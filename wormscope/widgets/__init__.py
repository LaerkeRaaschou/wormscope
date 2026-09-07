"""Reusable Qt widgets: the video canvas, the zoom/edit canvas and the plots."""

from __future__ import annotations

from .colors import PANEL_BG, VIEWPORT_COL
from .imaging import np_to_qimage
from .plot_panel import HAVE_MPL, PlotPanel
from .video_view import VideoView
from .zoom_view import ZoomView
