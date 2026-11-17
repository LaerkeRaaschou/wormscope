"""The Zoom canvas: zoomable / pannable, and where mask editing happens.

The widget only reports *what the user did* (paint here, stroke finished, SAM
point here, manual-length point added / moved / removed); it never touches the
masks or the measurements itself — the main window does that.

Manual length tool (MODE_MEASURE), on the selected worm's line:
    click                add a point at the end of the line
    click on the line    insert a point into that segment
    drag a point         move it
    Shift+click a point  remove it
    right / middle drag  pan (as in every tool)
"""

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..config import (BRUSH_DEFAULT, MANUAL_GRAB_PX, MANUAL_LINE_RGB,
                      MANUAL_OTHER_RGB, ZOOM_MAX, ZOOM_MIN)
from ..core.manual import nearest_point, nearest_segment
from .colors import PANEL_BG


class ZoomView(QWidget):
    paintRequested = pyqtSignal(float, float)
    strokeFinished = pyqtSignal()
    viewportChanged = pyqtSignal()
    pointAdded = pyqtSignal(float, float, int)   # x, y, label (1=worm, 0=not)
    # Manual length tool
    measureAdded = pyqtSignal(float, float, int)  # x, y, insert index (-1 = end)
    measureMoved = pyqtSignal(int, float, float)  # point index, x, y (dragging)
    measureRemoved = pyqtSignal(int)              # point index
    measureMoveFinished = pyqtSignal()            # drag released

    MODE_PAN, MODE_DRAW, MODE_ERASE = "pan", "draw", "erase"
    MODE_SAM_POS, MODE_SAM_NEG = "sam+", "sam-"
    MODE_MEASURE = "measure"

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
        # Manual length lines, set by the main window on every redraw:
        self.measure_active = []             # selected worm's points (editable)
        self.measure_others = []             # other worms' lines on this frame
        self.measure_label = ""              # length text next to the line
        self._dragging_pt = None             # index of the point being dragged
        self.setMinimumSize(240, 180)

    def set_measure(self, active, others=(), label=""):
        """The manual length lines to draw: the selected worm's (`active`,
        editable) and the other worms' on this frame (thinner, read-only)."""
        self.measure_active = list(active or [])
        self.measure_others = [list(p) for p in others]
        self.measure_label = label
        self.update()

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

    def pan_by(self, fx, fy):
        """Move the view by a fraction of its own width / height (the pan
        keys): 0.15 moves the picture 15 % of the view to the left."""
        if self._img is None or self.cx is None:
            return
        self.cx += fx * self.width() / self.zoom
        self.cy += fy * self.height() / self.zoom
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
        self._paint_measure(p, left, top)

    def _paint_measure(self, p, left, top):
        """The manual length lines, drawn crisp at screen resolution."""
        if not self.measure_active and not self.measure_others:
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        def to_w(pt):
            return QPointF(left + pt[0] * self.zoom, top + pt[1] * self.zoom)

        for pts in self.measure_others:
            p.setPen(QPen(QColor(*MANUAL_OTHER_RGB), 1.5))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(to_w(a), to_w(b))
        pts = self.measure_active
        if not pts:
            return
        col = QColor(*MANUAL_LINE_RGB)
        p.setPen(QPen(col, 2))
        for a, b in zip(pts, pts[1:]):
            p.drawLine(to_w(a), to_w(b))
        p.setPen(QPen(QColor(255, 255, 255), 1.2))
        for i, pt in enumerate(pts):
            p.setBrush(col if i else QColor(255, 255, 255))   # first point hollow-white
            p.drawEllipse(to_w(pt), 4.0, 4.0)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.measure_label:
            end = to_w(pts[-1])
            f = QFont(p.font())
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(0, 0, 0, 160))
            p.drawText(end + QPointF(9, -7), self.measure_label)
            p.setPen(col)
            p.drawText(end + QPointF(8, -8), self.measure_label)

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
        elif self.mode == self.MODE_MEASURE and e.button() == Qt.MouseButton.LeftButton:
            self._measure_press(e)
        else:
            self._panning = True
            self._last_pan = (e.position().x(), e.position().y())

    def _measure_press(self, e):
        """Left click with the manual length tool: grab / remove a point, insert
        one into a segment, or add one at the end."""
        pt = self._to_image(e)
        if pt is None:
            return
        radius = MANUAL_GRAB_PX / max(self.zoom, 1e-6)    # screen px -> image px
        hit = nearest_point(self.measure_active, pt[0], pt[1], radius)
        if hit is not None:
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.measureRemoved.emit(hit)
            else:
                self._dragging_pt = hit
            return
        seg = nearest_segment(self.measure_active, pt[0], pt[1], radius)
        self.measureAdded.emit(pt[0], pt[1], -1 if seg is None else seg + 1)

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
        if self._dragging_pt is not None:
            pt = self._to_image(e)
            if pt is not None:
                self.measureMoved.emit(self._dragging_pt, *pt)
        elif self._painting:
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
        if self._dragging_pt is not None:
            self._dragging_pt = None
            self.measureMoveFinished.emit()
        if self._painting:
            self._painting = False
            self._swap_paint = False
            self.strokeFinished.emit()
        self._panning = False
        self._last_pan = None
