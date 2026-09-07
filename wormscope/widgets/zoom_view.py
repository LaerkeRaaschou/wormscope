"""The Zoom canvas: zoomable / pannable, and where mask editing happens.

The widget only reports *what the user did* (paint here, stroke finished, SAM
point here); it never touches the masks itself — the main window does that.
"""

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QWidget

from ..config import BRUSH_DEFAULT, ZOOM_MAX, ZOOM_MIN
from .colors import PANEL_BG


class ZoomView(QWidget):
    paintRequested = pyqtSignal(float, float)
    strokeFinished = pyqtSignal()
    viewportChanged = pyqtSignal()
    pointAdded = pyqtSignal(float, float, int)   # x, y, label (1=worm, 0=not)

    MODE_PAN, MODE_DRAW, MODE_ERASE = "pan", "draw", "erase"
    MODE_SAM_POS, MODE_SAM_NEG = "sam+", "sam-"

    def __init__(self):
        super().__init__()
        self._img: QImage | None = None
        self._iw = self._ih = 1
        self.zoom = 4.0
        self.cx = None                       # image-space centre of the view
        self.cy = None
        self.mode = self.MODE_PAN
        self.brush = BRUSH_DEFAULT
        self._painting = False
        self._swap_paint = False             # Shift held: draw <-> erase
        self._panning = False
        self._last_pan = None
        self.setMinimumSize(240, 180)

    # --- image / view state --------------------------------------------------
    def set_image(self, qimg, iw, ih):
        first = self._img is None or self.cx is None
        self._img, self._iw, self._ih = qimg, iw, ih
        if first:                            # centre on first image
            self.cx, self.cy = iw / 2, ih / 2
        self.update()

    def has_image(self):
        return self._img is not None

    def _clamp(self):
        if self.cx is None:
            return
        self.cx = min(max(0.0, self.cx), self._iw)
        self.cy = min(max(0.0, self.cy), self._ih)

    def set_zoom(self, z):
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, z))
        self._clamp()
        self.update()
        self.viewportChanged.emit()

    def frame_bbox(self, bbox, margin=1.7):
        """Zoom/centre so the worm's bbox comfortably fills the view."""
        x0, y0, x1, y1 = bbox
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        W, H = max(1, self.width()), max(1, self.height())
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, min(W / (bw * margin),
                                                    H / (bh * margin))))
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self._clamp()
        self.update()
        self.viewportChanged.emit()

    def frame_fit(self):
        if self._img is None:
            return
        W, H = max(1, self.width()), max(1, self.height())
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, min(W / self._iw, H / self._ih)))
        self.cx, self.cy = self._iw / 2, self._ih / 2
        self.update()
        self.viewportChanged.emit()

    def _placement(self):
        return (self.width() / 2 - self.cx * self.zoom,
                self.height() / 2 - self.cy * self.zoom)

    def _to_image(self, e):
        if self._img is None or self.cx is None:
            return None
        left, top = self._placement()
        x = (e.position().x() - left) / self.zoom
        y = (e.position().y() - top) / self.zoom
        if 0 <= x < self._iw and 0 <= y < self._ih:
            return float(x), float(y)
        return None

    def viewport_rect(self):
        if self._img is None or self.cx is None:
            return None
        left, top = self._placement()
        W, H = self.width(), self.height()
        return (max(0.0, (0 - left) / self.zoom), max(0.0, (0 - top) / self.zoom),
                min(self._iw, (W - left) / self.zoom),
                min(self._ih, (H - top) / self.zoom))

    # --- events --------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), PANEL_BG)
        if self._img is None or self.cx is None:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Zoom")
            return
        left, top = self._placement()
        p.drawImage(QRectF(left, top, self._iw * self.zoom, self._ih * self.zoom),
                    self._img)

    def wheelEvent(self, e):
        if self._img is None:
            return
        pt = self._to_image(e)
        factor = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * factor))
        if pt is not None:                   # keep the point under the cursor put
            ex, ey = e.position().x(), e.position().y()
            self.cx = pt[0] + (self.width() / 2 - ex) / self.zoom
            self.cy = pt[1] + (self.height() / 2 - ey) / self.zoom
        self._clamp()
        self.update()
        self.viewportChanged.emit()

    def mousePressEvent(self, e):
        if self._img is None:
            return
        if self.mode in (self.MODE_DRAW, self.MODE_ERASE):
            pt = self._to_image(e)
            if pt is not None:
                # Decided once, at press: the whole stroke is one or the other,
                # so letting go of Shift halfway can't flip you mid-drag.
                self._swap_paint = bool(e.modifiers()
                                        & Qt.KeyboardModifier.ShiftModifier)
                self._painting = True
                self.paintRequested.emit(*pt)
        elif self.mode in (self.MODE_SAM_POS, self.MODE_SAM_NEG):
            pt = self._to_image(e)
            if pt is not None:
                self.pointAdded.emit(pt[0], pt[1], self._sam_label(e))
        else:
            self._panning = True
            self._last_pan = (e.position().x(), e.position().y())

    def paint_mode(self):
        """Draw or erase for the stroke in progress — **Shift swaps them**.

        The same gesture as the SAM tools: the toolbar holds the default and
        Shift borrows the opposite one for a single stroke, so tidying an edge
        you just over-painted costs no trip back to the buttons. Outside a
        Shift-stroke this is simply :attr:`mode`."""
        if self.mode not in (self.MODE_DRAW, self.MODE_ERASE):
            return self.mode
        if not self._swap_paint:
            return self.mode
        return self.MODE_ERASE if self.mode == self.MODE_DRAW else self.MODE_DRAW

    def _sam_label(self, e):
        """1 = worm, 0 = not worm.

        A plain click is positive; **Shift+click is negative**. That is the whole
        +/− switch — no need to go back to the toolbar between a point on the
        worm and one on something that isn't it. (MODE_SAM_NEG flips the default
        the other way round for anyone who sets it programmatically.)"""
        label = 0 if self.mode == self.MODE_SAM_NEG else 1
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            label = 1 - label
        return label

    def mouseMoveEvent(self, e):
        if self._painting:
            pt = self._to_image(e)
            if pt is not None:
                self.paintRequested.emit(*pt)
        elif self._panning and self._last_pan is not None:
            dx = e.position().x() - self._last_pan[0]
            dy = e.position().y() - self._last_pan[1]
            self._last_pan = (e.position().x(), e.position().y())
            self.cx -= dx / self.zoom
            self.cy -= dy / self.zoom
            self._clamp()
            self.update()
            self.viewportChanged.emit()

    def mouseReleaseEvent(self, e):
        if self._painting:
            self._painting = False
            self._swap_paint = False
            self.strokeFinished.emit()
        self._panning = False
        self._last_pan = None
