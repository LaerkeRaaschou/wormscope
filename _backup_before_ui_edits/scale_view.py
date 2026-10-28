"""The calibration dialog's canvas: draw a line along a scale bar.

Mouse (plus the wheel / middle-drag / double-click of ImageCanvas):
    left-drag            draw the line (a new drag replaces it)
    drag an end point    move that end
    Shift while dragging keep the line exactly horizontal or vertical
"""

import math

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPen

from ..config import SCALE_LINE_RGB
from .image_canvas import ImageCanvas

GRAB_PX = 8             # how close (on screen) counts as grabbing an end point


class ScaleView(ImageCanvas):
    lineChanged = pyqtSignal(float)        # length in image pixels (0 = none)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(560, 400)
        self.line = None                   # ((x0, y0), (x1, y1)) image pixels
        self._moving = None                # index of the end being dragged

    # --- the line ------------------------------------------------------------
    def length(self):
        if self.line is None:
            return 0.0
        (x0, y0), (x1, y1) = self.line
        return math.hypot(x1 - x0, y1 - y0)

    def set_line(self, line):
        self.line = None if line is None else (tuple(line[0]), tuple(line[1]))
        self.update()
        self.lineChanged.emit(self.length())

    def clear_image(self, message=""):
        super().clear_image(message)
        self.set_line(None)

    # --- painting ------------------------------------------------------------
    def paint_overlay(self, p):
        if self.line is None:
            return
        a = self._to_widget(*self.line[0])
        b = self._to_widget(*self.line[1])
        col = QColor(*SCALE_LINE_RGB)
        # dark halo first, so the line shows on light and dark images alike
        p.setPen(QPen(QColor(0, 0, 0, 160), 4))
        p.drawLine(a, b)
        p.setPen(QPen(col, 2))
        p.drawLine(a, b)
        # end ticks, perpendicular to the line
        d = b - a
        n = math.hypot(d.x(), d.y()) or 1.0
        perp = QPointF(-d.y() / n, d.x() / n) * 7
        for end in (a, b):
            p.drawLine(end - perp, end + perp)
        p.setPen(QColor(255, 255, 255))
        mid = (a + b) / 2 + perp * 2.2
        p.drawText(mid, f"{self.length():.1f} px")

    # --- mouse ---------------------------------------------------------------
    def _end_near(self, pos):
        if self.line is None:
            return None
        for i, end in enumerate(self.line):
            w = self._to_widget(*end)
            if math.hypot(w.x() - pos.x(), w.y() - pos.y()) <= GRAB_PX:
                return i
        return None

    def _snapped(self, fixed, moving, shift):
        """Keep the line horizontal or vertical (whichever is closer)."""
        if not shift:
            return moving
        if abs(moving[0] - fixed[0]) >= abs(moving[1] - fixed[1]):
            return (moving[0], fixed[1])
        return (fixed[0], moving[1])

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        if self._img is None or e.button() != Qt.MouseButton.LeftButton:
            return
        grabbed = self._end_near(e.position())
        if grabbed is not None:
            self._moving = grabbed
            return
        pt = self._to_image(e.position())
        if pt is not None:
            self.line = (pt, pt)
            self._moving = 1
            self.update()

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        if self._moving is None:
            hover = self._end_near(e.position()) is not None
            self.setCursor(Qt.CursorShape.SizeAllCursor if hover
                           else Qt.CursorShape.CrossCursor)
            return
        fixed = self.line[1 - self._moving]
        pt = self._to_image(e.position(), clamp=True)
        pt = self._snapped(fixed, pt,
                           bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier))
        ends = [None, None]
        ends[self._moving], ends[1 - self._moving] = pt, fixed
        self.line = (ends[0], ends[1])
        self.update()
        self.lineChanged.emit(self.length())

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if e.button() == Qt.MouseButton.LeftButton and self._moving is not None:
            self._moving = None
            if self.length() < 2:          # a click, not a line
                self.line = None
                self.update()
            self.lineChanged.emit(self.length())
