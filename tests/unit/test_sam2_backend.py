"""The SAM2 backend's edges — the parts that work without a model.

Loading the real thing needs torch, a multi-GB checkpoint and ideally a GPU, so
what is tested here is everything around it: how it reports missing files, how
it prepares the frames SAM2 wants, and the protocol the GUI depends on. The
behaviour *with* a model is covered by the workflow tests, driven against a
stand-in backend.
"""

import numpy as np
import pytest
from PIL import Image

from wormscope import segmentation
from wormscope.segmentation import (MissingSam2Paths, Sam2Backend,
                                    prepare_jpeg_dir)


def test_missing_paths_raise_a_named_error(monkeypatch, tmp_path):
    """A distinct exception type, not a generic RuntimeError: the GUI answers
    this one by opening the file picker rather than showing an error."""
    monkeypatch.setenv("WORMSCOPE_SETTINGS", str(tmp_path / "settings.json"))
    for var in ("SAM2_CFG", "SAM2_CKPT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(segmentation.sam2_backend.config, "SAM2_CFG", "")
    monkeypatch.setattr(segmentation.sam2_backend.config, "SAM2_CKPT", "")

    frames = [np.zeros((8, 8, 3), np.uint8)]
    with pytest.raises(MissingSam2Paths):
        Sam2Backend(frames, (8, 8))


def test_missing_paths_is_a_runtime_error_underneath():
    """So a caller that only knows about RuntimeError still catches it."""
    assert issubclass(MissingSam2Paths, RuntimeError)


def test_frames_are_prepared_as_jpegs_for_sam2(tmp_path):
    """SAM2's video predictor wants a folder of numbered JPEGs."""
    from wormscope.core.models import FrameSource
    src = FrameSource([np.full((12, 16, 3), 90, np.uint8) for _ in range(3)],
                      lambda a: a)
    folder = prepare_jpeg_dir(src)
    files = sorted(p.name for p in folder.iterdir())
    assert len(files) == 3
    assert all(f.lower().endswith((".jpg", ".jpeg")) for f in files), files
    assert files == sorted(files), "numbered so they sort into frame order"
    img = Image.open(folder / files[0])
    assert img.size == (16, 12)


def test_the_factory_is_the_swap_point(monkeypatch):
    """Everything the GUI does goes through make_segmenter, so replacing the
    model is one assignment."""
    class Other:
        def __init__(self, frames, shape):
            self.shape = shape

    monkeypatch.setattr(segmentation, "SEGMENTER_FACTORY", Other)
    made = segmentation.make_segmenter([np.zeros((4, 4, 3), np.uint8)], (4, 4))
    assert isinstance(made, Other) and made.shape == (4, 4)


def test_the_default_factory_is_sam2():
    assert segmentation.SEGMENTER_FACTORY is Sam2Backend
