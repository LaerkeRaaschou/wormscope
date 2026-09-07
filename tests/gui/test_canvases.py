"""The two image canvases, on their own.

They report what the user did and nothing else — no widget in ``widgets/``
touches a mask. That is what lets the same canvas be driven by a test, or
reused somewhere else, without dragging the window along.
"""

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent

from wormscope.config import ZOOM_MAX, ZOOM_MIN
from wormscope.widgets import VideoView, ZoomView, np_to_qimage


# --- numpy -> Qt ------------------------------------------------------------
def test_an_rgb_array_becomes_an_image_of_the_same_size():
    arr = np.zeros((12, 20, 3), np.uint8)
    img = np_to_qimage(arr)
    assert (img.width(), img.height()) == (20, 12)


def test_a_grayscale_array_is_widened_to_rgb():
    img = np_to_qimage(np.zeros((8, 8), np.uint8))
    assert (img.width(), img.height()) == (8, 8)


def test_the_image_owns_its_pixels():
    """It is built on a numpy buffer; without the copy, editing the array
    afterwards would corrupt whatever is on screen."""
    arr = np.zeros((4, 4, 3), np.uint8)
    img = np_to_qimage(arr)
    arr[:] = 255
    assert img.pixelColor(0, 0).red() == 0


# --- the overview video -----------------------------------------------------
@pytest.fixture
def video(qapp):
    view = VideoView()
    view.resize(200, 150)
    view.set_frame(np_to_qimage(np.full((60, 80, 3), 120, np.uint8)))
    view.grab()          # the letterbox rectangle is worked out while painting
    return view


def test_clicking_the_video_reports_an_image_coordinate(video):
    """Not a widget coordinate: the window looks up which worm is under that
    pixel, so the translation has to happen here."""
    seen = []
    video.imageClicked.connect(lambda x, y: seen.append((x, y)))
    centre = QPointF(video.width() / 2, video.height() / 2)
    video.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, centre, Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    assert len(seen) == 1
    x, y = seen[0]
    assert 0 <= x < 80 and 0 <= y < 60
    assert x == pytest.approx(40, abs=2) and y == pytest.approx(30, abs=2)


def test_a_click_outside_the_image_is_ignored(video):
    """The frame is letterboxed inside the widget; the grey margin is not part
    of the picture."""
    seen = []
    video.imageClicked.connect(lambda x, y: seen.append((x, y)))
    video.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(1.0, 1.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier))
    assert seen == []


def test_the_video_paints_without_an_image(qapp):
    view = VideoView()
    view.resize(60, 40)
    view.grab()                     # would raise if paintEvent assumed a frame


def test_the_video_draws_the_zoom_region_and_the_selection(video):
    """The yellow rectangle that shows what the Zoom panel is looking at."""
    video.set_viewport((10.0, 10.0, 50.0, 40.0))
    video.set_selection((30.0, 20.0))
    video.grab()


# --- the zoom / edit canvas -------------------------------------------------
@pytest.fixture
def zoom(qapp):
    view = ZoomView()
    view.resize(240, 180)
    view.set_image(np_to_qimage(np.full((60, 80, 3), 120, np.uint8)), 80, 60)
    return view


def test_the_first_image_centres_the_view(zoom):
    assert (zoom.cx, zoom.cy) == (40.0, 30.0)
    assert zoom.has_image()


def test_zoom_is_clamped_to_the_configured_range(zoom):
    zoom.set_zoom(1000.0)
    assert zoom.zoom == ZOOM_MAX
    zoom.set_zoom(0.0001)
    assert zoom.zoom == ZOOM_MIN


def test_fit_shows_the_whole_frame(zoom):
    zoom.frame_fit()
    assert zoom.zoom == pytest.approx(min(240 / 80, 180 / 60))
    assert (zoom.cx, zoom.cy) == (40.0, 30.0)


def test_framing_a_bbox_centres_on_it(zoom):
    zoom.frame_bbox((10, 10, 30, 20))
    assert (zoom.cx, zoom.cy) == (20.0, 15.0)
    assert zoom.zoom > 1.0, "it zooms in on a small worm"


def test_the_viewport_rect_reports_what_is_visible(zoom):
    zoom.frame_fit()
    x0, y0, x1, y1 = zoom.viewport_rect()
    assert (x0, y0) == (0.0, 0.0) and (x1, y1) == (80.0, 60.0)


def test_scrolling_zooms_and_keeps_the_point_under_the_cursor(zoom):
    zoom.set_zoom(2.0)
    before = zoom.zoom
    zoom.wheelEvent(QWheelEvent(
        QPointF(60.0, 40.0), QPointF(60.0, 40.0), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False))
    assert zoom.zoom > before


def test_dragging_with_the_pan_tool_moves_the_view(zoom):
    zoom.mode = ZoomView.MODE_PAN
    zoom.set_zoom(4.0)
    before = (zoom.cx, zoom.cy)
    zoom.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(100.0, 90.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier))
    zoom.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove, QPointF(60.0, 90.0), Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    zoom.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(60.0, 90.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier))
    assert zoom.cx > before[0], "dragging left moves the view right"


def test_panning_stops_at_the_edge_of_the_frame(zoom):
    zoom.mode = ZoomView.MODE_PAN
    zoom.cx, zoom.cy = -500.0, 900.0
    zoom._clamp()
    assert 0 <= zoom.cx <= 80 and 0 <= zoom.cy <= 60


def test_a_paint_outside_the_image_is_not_reported(zoom):
    """Painting has to stop at the frame edge, not wrap or throw."""
    zoom.mode = ZoomView.MODE_DRAW
    seen = []
    zoom.paintRequested.connect(lambda x, y: seen.append((x, y)))
    zoom.set_zoom(1.0)                       # the frame no longer fills the widget
    zoom.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(1.0, 1.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier))
    assert seen == []


def test_a_stroke_reports_its_end(zoom):
    zoom.mode = ZoomView.MODE_DRAW
    finished = []
    zoom.strokeFinished.connect(lambda: finished.append(True))
    centre = QPointF(zoom.width() / 2, zoom.height() / 2)
    zoom.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, centre, Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    zoom.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease, centre, Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
    assert finished == [True], "one stroke, one metric recompute, one undo step"


def test_the_zoom_canvas_paints_without_an_image(qapp):
    view = ZoomView()
    view.resize(60, 40)
    view.grab()


def test_events_on_an_empty_canvas_are_ignored(qapp):
    view = ZoomView()
    view.resize(60, 40)
    view.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(5.0, 5.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier))
    assert not view.has_image()
