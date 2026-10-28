"""Per-worm metric bookkeeping shared by the GUI and by batch scripts."""

from ..config import ACCEL_METRIC, SPEED_METRIC
from .geometry import mask_stats
from .kinematics import refresh_kinematics


def apply_mask_stats(worm, masks, idx, area_metric, length_metric):
    """Recompute area / length / centroid / bbox for one worm-frame from its
    mask (no kinematics — callers do that once per worm)."""
    mask = masks.get(worm.obj_id, idx)
    area, length, _skel, centroid, bbox = mask_stats(mask)
    worm.metrics.setdefault(area_metric, {})[idx] = area
    if length is not None:
        worm.metrics.setdefault(length_metric, {})[idx] = length
    elif idx in worm.metrics.get(length_metric, {}):
        worm.metrics[length_metric][idx] = float("nan")
    if centroid is None:
        worm.centroids.pop(idx, None)
        worm.bboxes.pop(idx, None)
    else:
        worm.centroids[idx] = centroid
        worm.bboxes[idx] = bbox


def recompute_worm(worm, masks, area_metric, length_metric, frames=None):
    """Recompute every metric for one worm. `frames` defaults to every frame the
    worm has a mask for."""
    for idx in (masks.frames_of(worm.obj_id) if frames is None else frames):
        apply_mask_stats(worm, masks, idx, area_metric, length_metric)
    refresh_kinematics(worm)


# Lengths the scale tool accepts, in µm. Everything is stored as µm per pixel.
UNIT_TO_UM = {"nm": 1e-3, "µm": 1.0, "mm": 1e3}


def um_per_px(line_px, known_length, unit="µm"):
    """Calibration from a line drawn on a scale bar: µm per pixel, or None.

    `line_px` is the drawn line's length in pixels, `known_length` the real
    length it spans, in `unit` (a key of UNIT_TO_UM)."""
    if not line_px or not known_length or line_px <= 0 or known_length <= 0:
        return None
    return known_length * UNIT_TO_UM[unit] / line_px


def unit_scales(area_metric, length_metric, px_to_um=None, fps=None):
    """metric name -> (factor, label). Empty (i.e. pixels) until calibrated."""
    if not px_to_um:
        return {}
    u = px_to_um
    scales = {area_metric: (u * u, "area (µm²)"),
              length_metric: (u, "length (µm)")}
    if fps:
        scales[SPEED_METRIC] = (u * fps, "speed (µm/s)")
        scales[ACCEL_METRIC] = (u * fps * fps, "accel (µm/s²)")
    else:
        scales[SPEED_METRIC] = (u, "speed (µm/frame)")
        scales[ACCEL_METRIC] = (u, "accel (µm/frame²)")
    return scales


def plot_specs(area_metric, length_metric, px_to_um=None, fps=None):
    """(name, label, factor) for the four metrics, honouring calibration."""
    scales = unit_scales(area_metric, length_metric, px_to_um, fps)
    specs = []
    for name in (area_metric, length_metric, SPEED_METRIC, ACCEL_METRIC):
        factor, label = scales.get(name, (1.0, name))
        specs.append((name, label, factor))
    return specs
