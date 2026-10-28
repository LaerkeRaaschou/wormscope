"""A zoomable image canvas to build tool canvases on (CountGD, scale bar…).

It owns the image, the zoom/pan state and the image <-> widget coordinate
mapping; subclasses add what is drawn on top (``paint_overlay``) and what the
left / right mouse buttons do. Common to all of them:

    wheel          zoom around the cursor
    right-drag     pan (middle-drag too; left-drag as well when `pan_mode`)
    double-click   fit the whole image

``zoom_by`` / ``fit`` are there for buttons, and ``zoomChanged`` reports the
magnification (1.0 = one image pixel per screen pixel) so a label can follow.
"""

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QWidget

from ..config import PANEL_BG_RGB

ZOOM_LIMITS = (1.0, 40.0)


class ImageCanvas(QWidget):
    zoomChanged = pyqtSignal(float)        # screen px per image px

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._img = None
        self._iw = self._ih = 0
        self._zf = 1.0                     # zoom relative to "fit"
        self._cx = self._cy = 0.0          # image point at the widget centre
        self._pan = None                   # last widget position while panning
        self.pan_mode = False              # True: left-drag pans too (Pan tool)
        self.message = ""

    # --- content -------------------------------------------------------------
    def set_image(self, qimg, iw, ih):
        resized = (iw, ih) != (self._iw, self._ih)
        self._img, self._iw, self._ih = qimg, iw, ih
        if resized:
            self.fit()
        self.update()

    def clear_image(self, message=""):
        self._img, self._iw, self._ih = None, 0, 0
        self.message = message
        self.update()

    def has_image(self):
        return self._img is not None

    def fit(self):
        self._zf = 1.0
        self._cx, self._cy = self._iw / 2, self._ih / 2
        self.update()
        self.zoomChanged.emit(self.magnification())

    def magnification(self):
        """Screen pixels per image pixel (1.0 = 100 %)."""
        return self._scale() if self._iw else 1.0

    def zoom_by(self, factor):
        """Zoom around the centre of the view (the − / + buttons)."""
        if self._img is None:
            return
        lo, hi = ZOOM_LIMITS
        self._zf = min(hi, max(lo, self._zf * factor))
        if self._zf == 1.0:
            self._cx, self._cy = self._iw / 2, self._ih / 2
        self._clamp_centre()
        self.update()
        self.zoomChanged.emit(self.magnification())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.zoomChanged.emit(self.magnification())

    # --- geometry ------------------------------------------------------------
    def _scale(self):
        if not self._iw:
            return 1.0
        fit = min(self.width() / self._iw, self.height() / self._ih)
        return fit * self._zf

    def _to_widget(self, x, y):
        s = self._scale()
        return QPointF(self.width() / 2 + (x - self._cx) * s,
                       self.height() / 2 + (y - self._cy) * s)

    def _to_image(self, pos, clamp=False):
        """Widget position -> image (x, y); None outside unless `clamp`."""
        s = self._scale()
        x = self._cx + (pos.x() - self.width() / 2) / s
        y = self._cy + (pos.y() - self.height() / 2) / s
        if clamp:
            return min(max(x, 0.0), self._iw), min(max(y, 0.0), self._ih)
        if 0 <= x < self._iw and 0 <= y < self._ih:
            return x, y
        return None

    def _rect(self, x0, y0, x1, y1):
        return QRectF(self._to_widget(x0, y0), self._to_widget(x1, y1))

    def _clamp_centre(self):
        self._cx = min(max(self._cx, 0.0), float(self._iw))
        self._cy = min(max(self._cy, 0.0), float(self._ih))

    # --- painting ------------------------------------------------------------
    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(*PANEL_BG_RGB))
        if self._img is None:
            p.setPen(QColor(60, 70, 90))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zf < 3)
        p.drawImage(self._rect(0, 0, self._iw, self._ih), self._img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_overlay(p)

    def paint_overlay(self, painter):
        """Draw on top of the image (image coordinates via _to_widget)."""

    # --- mouse: pan / zoom / fit (subclasses handle left and right) ----------
    def _is_pan_press(self, e):
        """Middle or right button always pan; the left one only in pan_mode."""
        b = e.button()
        return (b in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton)
                or (self.pan_mode and b == Qt.MouseButton.LeftButton))

    def mousePressEvent(self, e):
        if self._img is not None and self._is_pan_press(e):
            self._pan = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            s = self._scale()
            self._cx -= d.x() / s
            self._cy -= d.y() / s
            self._clamp_centre()
            self.update()

    def mouseReleaseEvent(self, e):
        if self._pan is not None and self._is_pan_press(e):
            self._pan = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.pan_mode
                           else Qt.CursorShape.CrossCursor)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.fit()

    def wheelEvent(self, e):
        if self._img is None:
            return
        steps = e.angleDelta().y() / 120
        if not steps:
            return
        anchor = self._to_image(e.position(), clamp=True)
        lo, hi = ZOOM_LIMITS
        self._zf = min(hi, max(lo, self._zf * 1.25 ** steps))
        s = self._scale()
        # keep the image point under the cursor where it was
        self._cx = anchor[0] - (e.position().x() - self.width() / 2) / s
        self._cy = anchor[1] - (e.position().y() - self.height() / 2) / s
        if self._zf == 1.0:
            self._cx, self._cy = self._iw / 2, self._ih / 2
        self._clamp_centre()
        self.update()
        self.zoomChanged.emit(self.magnification())
