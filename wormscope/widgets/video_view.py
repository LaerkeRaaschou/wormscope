"""The Video canvas: whole frame + masks, with the zoom viewport rectangle."""

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from .colors import PANEL_BG, VIEWPORT_COL


class VideoView(QWidget):
    """Displays the composited frame, letterboxed, and emits image-space
    coordinates when clicked."""

    imageClicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self._img: QImage | None = None
        self._sel = None
        self._viewport = None                # (x0, y0, x1, y1) in image coords
        self._rect: QRectF | None = None
        self._scale = 1.0
        self.setMinimumSize(360, 240)

    def set_frame(self, qimg):
        self._img = qimg
        self.update()

    def set_selection(self, centroid):
        self._sel = centroid
        self.update()

    def set_viewport(self, rect):
        self._viewport = rect
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), PANEL_BG)
        if self._img is None:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Video")
            return
        iw, ih = self._img.width(), self._img.height()
        self._scale = min(self.width() / iw, self.height() / ih)
        dw, dh = iw * self._scale, ih * self._scale
        ox, oy = (self.width() - dw) / 2, (self.height() - dh) / 2
        self._rect = QRectF(ox, oy, dw, dh)
        p.drawImage(self._rect, self._img)
        if self._viewport is not None:
            x0, y0, x1, y1 = self._viewport
            p.setPen(QPen(VIEWPORT_COL, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(ox + x0 * self._scale, oy + y0 * self._scale,
                              (x1 - x0) * self._scale, (y1 - y0) * self._scale))

    def mousePressEvent(self, e):
        if self._img is None or self._rect is None:
            return
        x = (e.position().x() - self._rect.x()) / self._scale
        y = (e.position().y() - self._rect.y()) / self._scale
        if 0 <= x < self._img.width() and 0 <= y < self._img.height():
            self.imageClicked.emit(float(x), float(y))
