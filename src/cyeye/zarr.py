import os
import posixpath
import numpy as np
import ngff_zarr as nz
import zarr
import zarr.storage
from zarr.abc.store import Store
from dataclasses import dataclass, field
from typing import Iterator, Union, Sequence
import fsspec

type PathLike = os.PathLike[str] | str

class UnsetType:
    pass

UNSET = UnsetType()

type SelectionIndex = Union[
    int,                # single index
    tuple[int, int],    # range [start, stop)
    Sequence[int],      # multiple indices
    UnsetType           # full slice (:)
]

def to_array_index(index: SelectionIndex) -> int | slice | Sequence[int]:
    """Converts a `SelectionIndex` to the index used to slice an array"""
    if isinstance(index, UnsetType):
        return slice(None)
    if isinstance(index, tuple) and len(index) == 2:
        start, stop = index
        return slice(start, stop)
    return index

type WellIndex = tuple[str, str]  # (row name, column name), e.g. ("A", "1")

@dataclass
class ZarrROI:
    """Defines a region of interest in an OME-Zarr HCS plate"""
    time: SelectionIndex = UNSET
    channel: SelectionIndex = UNSET
    z: SelectionIndex = UNSET
    well: WellIndex | None = None
    field: int | None = None

@dataclass(kw_only=True)
class ZarrDataSelection(ZarrROI):
    """A region of interest bound to a specific plate, ready to load data"""
    plate: nz.HCSPlate

    _data_memo: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    ) # excluded from dataclass constructor

    def _get_well(self) -> nz.HCSWell:
        """Returns the currently selected `HCSWell`"""
        if self.well is None:
            raise ValueError("No well specified")
        row_name, column_name = self.well
        well = self.plate.get_well(row_name, column_name)
        if well is None:
            raise ValueError(f"Well '{row_name}/{column_name}' not found")
        return well

    def _get_field(self) -> nz.NgffMultiscales:
        """
        Returns the `NgffImage` at the currently selected resolution level
        """
        if self.field is None:
            raise ValueError("No field specified")
        field = self._get_well().get_image(self.field)
        if field is None:
            raise ValueError(f"Field {self.field} not found")
        return field

    def load(self) -> np.ndarray:
        # Return memoized result if available
        if self._data_memo is not None:
            return self._data_memo

        field = self._get_field()
        image = field.images[0]     # highest resolution

        # Return subarray
        data = image.data[
            to_array_index(self.time),
            to_array_index(self.channel),
            to_array_index(self.z)
        ]
        data = np.asarray(data)
        self._data_memo = data
        return data

    def iter_images(self) -> Iterator[np.ndarray]:
        """
        Iterates over each 2D (y, x) image within the selected region
        """
        data = self.load()
        images = data.reshape(-1, data.shape[-2], data.shape[-1])
        for image in images:
            yield image

@dataclass
class ZarrFieldShape:
    t: int
    c: int
    z: int
    y: int
    x: int

@dataclass
class Axes:
    time: nz.Axis
    channel: nz.Axis
    z: nz.Axis
    y: nz.Axis
    x: nz.Axis

def store_location(store) -> str:
    """
    Returns a location string (local path or URL) describing the root of
    `store`, i.e. whatever was originally passed to `ZarrDataManager()`.
    """
    if isinstance(store, zarr.storage.FsspecStore):
        protocol = store.fs.protocol
        scheme = protocol[0] \
            if isinstance(protocol, (list, tuple)) \
            else protocol
        return f"{scheme}://{store.path}"
    
    if isinstance(store, zarr.storage.LocalStore):
        return str(store.root)
    return os.fspath(store)

def sibling_store(store, name: str) -> Store:
    """
    Opens a writable Zarr store named `name`, located next to `store`'s root
    (i.e. as a sibling directory/key at the same parent location).
    """
    if isinstance(store, zarr.storage.FsspecStore):
        parent = posixpath.dirname(store.path.rstrip("/"))
        return zarr.storage.FsspecStore(
            fs=store.fs, path=posixpath.join(parent, name)
        )
    if isinstance(store, zarr.storage.LocalStore):
        root = store.root
    else:
        root = os.fspath(store)


    root = store.root if isinstance(store, zarr.storage.LocalStore) else os.fspath(store)
    parent = os.path.dirname(os.path.normpath(root))
    return zarr.storage.LocalStore(os.path.join(parent, name))

def sibling_name(store, suffix: str) -> str:
    """
    Derives a sibling dataset name from `store`'s root, e.g. a store rooted
    at ".../my-plate.zarr" with suffix "labels" becomes "my-plate_labels.zarr"
    """
    location = store_location(store)
    base = posixpath.basename(location.rstrip("/"))
    stem = base[: -len(".zarr")] if base.endswith(".zarr") else base
    return f"{stem}_{suffix}.zarr"

class ZarrDataManager:
    plate: nz.HCSPlate

    def __init__(self, store) -> None:
        self.plate = nz.from_hcs_zarr(store)

    @classmethod
    def download(
        cls,
        url: str,
        path: PathLike = os.getcwd()
    ) -> "ZarrDataManager":
        fs, root = fsspec.core.url_to_fs(url)
        fs.get(root, str(path), recursive=True)
        return cls(path)

    @property
    def name(self) -> str:
        return self.plate.metadata.name or ""

    @property
    def store(self):
        """The store this manager was opened from"""
        return self.plate.store

    @property
    def location(self) -> str:
        """The local path or URL this manager was opened from"""
        return store_location(self.store)

    def sibling_store(self, suffix: str) -> Store:
        """
        Opens a writable Zarr store located next to this dataset, e.g. for a
        dataset at ".../my-plate.zarr", `sibling_store("labels")` opens
        ".../my-plate_labels.zarr"
        """
        return sibling_store(self.store, sibling_name(self.store, suffix))

    @property
    def n_rows(self) -> int:
        return len(self.plate.rows)

    @property
    def n_columns(self) -> int:
        return len(self.plate.columns)

    @property
    def n_wells(self) -> int:
        return len(self.plate.metadata.wells)

    @property
    def n_fields(self) -> int | None:
        return self.plate.field_count

    def _reference_field(self) -> nz.NgffMultiscales:
        """
        Returns the first available field, used to read shared metadata
        """
        if not self.plate.wells:
            raise ValueError("Plate has no wells")
        
        first_well = self.plate.wells[0]
        well = self.plate.get_well_by_indices(
            first_well.rowIndex,
            first_well.columnIndex
        )
        if well is None or not well.images:
            raise ValueError("No fields found in plate")
        
        field = well.get_image(0)
        if field is None:
            raise ValueError("No fields found in plate")
        return field

    def dim_size(self, dim: str) -> int:
        image = self._reference_field().images[0]  # highest resolution
        if dim not in image.dims:
            return 1
        return image.data.shape[list(image.dims).index(dim)]

    @property
    def n_channels(self) -> int:
        return self.dim_size("c")

    @property
    def n_timepoints(self) -> int:
        return self.dim_size("t")

    @property
    def n_z_planes(self) -> int:
        return self.dim_size("z")

    @property
    def channel_names(self) -> list[str]:
        field = self._reference_field()

        omero = field.metadata.omero
        if omero:
            return [str(channel.label) for channel in omero.channels]

        return [str(i) for i in range(self.n_channels)]

    @property
    def axes(self) -> Axes:
        metadata = self._reference_field().metadata
        coordinate_systems = getattr(metadata, "coordinateSystems", None)

        if coordinate_systems: # ngff v0.6
            axes_list = coordinate_systems[0].axes
        else:   # v0.4/v0.5
            axes_list = getattr(metadata, 'axes', [])
        
        axes_by_name = { axis.name: axis for axis in axes_list }

        return Axes(
            time=axes_by_name['t'],
            channel=axes_by_name['c'],
            z=axes_by_name['z'],
            y=axes_by_name['y'],
            x=axes_by_name['x'],
        )

    def print_wells(self) -> None:
        """Prints a grid visualization of the wells that contain data"""
        rows = self.plate.rows
        columns = self.plate.columns
        occupied = [
            (well.rowIndex, well.columnIndex) for well in self.plate.wells
        ]
        well_char = "\u25a0"    # ■

        # print header (column labels)
        print((" ")*2, end="")
        for col in columns:
            print(col.name, end=" ")
        print()

        for row_index, row in enumerate(rows):
            # row label
            print(row.name, end=" ")

            # mark occupied wells
            for col_index, col in enumerate(columns):
                # pad entry to match width of column label
                format = lambda char: f"{char:<{len(col.name)}}"

                if (row_index, col_index) in occupied:
                    print(format(well_char), end=" ")
                else:
                    print(format(" "), end=" ")

            print() # end of row

    def select(
        self,
        well: WellIndex | None = None,
        field: int | None = None,
        time: SelectionIndex | None = None,
        channel: SelectionIndex | None = None,
        z: SelectionIndex | None = None,
    ) -> ZarrDataSelection:
        overrides = {
            "well": well,
            "field": field,
            "time": time,
            "channel": channel,
            "z": z,
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}
        return ZarrDataSelection(plate=self.plate, **overrides)
