"""numpy <-> Qt image conversion."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage


def np_to_qimage(arr: np.ndarray) -> QImage:
    """RGB (or grayscale) uint8 array -> QImage. Copies, so the caller may keep
    editing the array afterwards."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
