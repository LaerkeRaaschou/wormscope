from PyQt6.QtGui import QColor

from ..config import PANEL_BG_RGB, VIEWPORT_RGB

PANEL_BG = QColor(*PANEL_BG_RGB)        # letterbox around a fitted image
VIEWPORT_COL = QColor(*VIEWPORT_RGB)    # zoom-region rectangle on the video
