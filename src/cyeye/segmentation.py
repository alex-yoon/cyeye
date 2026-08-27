import zarr
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from cellpose import models

from .zarr import (
    UNSET,
    SelectionIndex,
    WellIndex,
    ZarrDataManager,
    resolve_indices,
    store_location,
)

Z_AXIS_TRANSFORM = "max"

@dataclass
class SegmentationResult:
    """The output of a Cellpose-SAM segmentation run over one well/field"""
    masks: np.ndarray          # (t, y, x); one 2D label mask per timepoint
    timepoints: list[int]      # absolute timepoint index per mask, aligned to masks' t axis
    channel_indices: list[int]
    channel_names: list[str]
    layer_name: str
    well: WellIndex
    field: int

def _load_projected_image(
    manager: ZarrDataManager,
    well: WellIndex,
    field: int,
    time: int,
    channel_indices: list[int],
) -> np.ndarray:
    """Loads all Z-planes for one timepoint/channel set, max-projected along Z"""
    selection = manager.select(
        well=well, field=field, time=time, channel=channel_indices
    )
    data = selection.load()            # (c, z, y, x) -- t squeezed by the int index
    projected = np.max(data, axis=1)   # (c, y, x)
    if len(channel_indices) == 1:
        return projected[0]            # (y, x)
    return projected

def run_cellpose_sam(
    manager: ZarrDataManager,
    *,
    well: WellIndex,
    field: int,
    layer_name: str,
    channels: SelectionIndex = UNSET,
    time: SelectionIndex = UNSET,
    model=None,
    eval_kwargs: dict[str, Any] | None = None,
) -> SegmentationResult:
    """
    Segments one well/field with Cellpose-SAM, one mask per timepoint, always
    max-projecting the Z-axis first.
    """

    eval_kwargs = dict(eval_kwargs or {})

    channel_indices = resolve_indices(channels, manager.n_channels)
    channel_names = [manager.channel_names[i] for i in channel_indices]
    timepoints = resolve_indices(time, manager.n_timepoints)

    if model is None:
        model = models.CellposeModel()

    # Projected image is (c, y, x) for multiple channels; tell Cellpose-SAM
    # where the channel axis is unless the caller already specified one.
    if len(channel_indices) > 1:
        eval_kwargs.setdefault("channel_axis", 0)

    masks = []
    for t in timepoints:
        image = _load_projected_image(manager, well, field, t, channel_indices)
        mask, _flows, _styles = model.eval(image, **eval_kwargs)
        masks.append(np.asarray(mask))

    return SegmentationResult(
        masks=np.stack(masks, axis=0),
        timepoints=timepoints,
        channel_indices=channel_indices,
        channel_names=channel_names,
        layer_name=layer_name,
        well=well,
        field=field,
    )

def save_segmentation(
    manager: ZarrDataManager,
    result: SegmentationResult,
    eval_kwargs: dict[str, Any] | None = None,
) -> str:
    """
    Writes `result` to a new OME-Zarr dataset next to the original dataset,
    under "<row>/<col>/<field>/labels/<layer_name>". Returns its location.
    """
    store = manager.sibling_store("labels")
    root = zarr.open_group(store, mode="a")

    row, col = result.well
    field_group = root.require_group(f"{row}/{col}/{result.field}")
    labels_group = field_group.require_group("labels")

    existing_labels = labels_group.attrs.get("ome", {}).get("labels", [])
    if result.layer_name not in existing_labels:
        labels_group.attrs["ome"] = {
            "version": "0.5",
            "labels": [*existing_labels, result.layer_name],
        }

    layer_group = labels_group.require_group(result.layer_name)
    layer_group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "name": result.layer_name,
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                ],
            }],
        }],
        "image-label": {
            "source": {
                "image": f"{manager.location}/{row}/{col}/{result.field}",
            },
        },
    }
    layer_group.attrs["cyeye"] = {
        "z-axis-transform": Z_AXIS_TRANSFORM,
        "timepoints": result.timepoints,
        "channels": {
            "indices": result.channel_indices,
            "names": result.channel_names,
        },
        "eval_kwargs": eval_kwargs or {},
        "created": datetime.now(timezone.utc).isoformat(),
    }

    layer_group.create_array(
        "0",
        data=result.masks,
        dimension_names=["t", "y", "x"],
        overwrite=True,
    )

    return store_location(store)

def segment(
    manager: ZarrDataManager,
    *,
    well: WellIndex,
    field: int,
    layer_name: str,
    channels: SelectionIndex = UNSET,
    time: SelectionIndex = UNSET,
    model=None,
    eval_kwargs: dict[str, Any] | None = None,
) -> SegmentationResult:
    """
    Segments cells with Cellpose-SAM and persists the resulting masks next to
    the original dataset. Returns the predicted masks and their metadata.
    """
    result = run_cellpose_sam(
        manager,
        well=well,
        field=field,
        layer_name=layer_name,
        channels=channels,
        time=time,
        model=model,
        eval_kwargs=eval_kwargs,
    )
    save_segmentation(manager, result, eval_kwargs=eval_kwargs)
    return result
