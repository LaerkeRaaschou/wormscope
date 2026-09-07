"""Pluggable segmentation backends.

``SEGMENTER_FACTORY`` is called as ``factory(frames, (h, w))`` and must return an
object with ``add_points(frame_idx, obj_id, points, labels)``, ``propagate()``
and ``reset()``. Override it to swap in another model or a test stand-in::

    from wormscope import segmentation
    segmentation.SEGMENTER_FACTORY = MyBackend

The SAM2 config + checkpoint paths come from :mod:`wormscope.settings`, which
remembers them between runs.
"""

from __future__ import annotations

from .sam2_backend import (MissingSam2Paths, Sam2Backend,  # noqa: F401
                           hydra_config_name, pair_warning, prepare_jpeg_dir,
                           sam2_config_dir, sam2_package_dir)

SEGMENTER_FACTORY = Sam2Backend


def make_segmenter(frames, shape):
    """Build a segmenter with the currently-installed factory."""
    return SEGMENTER_FACTORY(frames, shape)
