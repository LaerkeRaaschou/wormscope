"""The inline progress panel.

Never a modal dialog: long jobs report in the corner of the Zoom panel's
control column while the rest of the window stays usable. It shows a
percentage rather than a count because the frame total is an estimate — a run
can stop early, and a worm's prompts need not span the whole clip — so a count
can march past its own total.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame

from wormscope.ui.progress import Progress, ProgressPanel


def test_the_panel_is_hidden_until_something_runs(window):
    assert window.progress_panel.isHidden()


def test_starting_shows_it_with_what_it_is_doing(window):
    panel = window.progress_panel
    panel.start("Segmenting", total=98, detail="worm 1")
    assert not panel.isHidden()
    assert panel.status.toolTip() == "Segmenting — worm 1"
    panel.finish()


def test_progress_is_reported_as_a_percentage(window):
    panel = window.progress_panel
    panel.start("Segmenting", total=98, detail="worm 1")
    panel.advance(49)
    assert panel._count == "50%"
    assert panel.status.toolTip() == "Segmenting 50% — worm 1"
    assert "49" not in panel.status.toolTip(), "no frame count to read past"
    panel.finish()


def test_it_never_claims_more_than_a_hundred_percent(window):
    panel = window.progress_panel
    panel.start("Segmenting", total=98)
    panel.advance(120)                       # a run can overshoot its estimate
    assert panel._count == "100%"
    assert panel.bar.value() == panel.bar.maximum()
    panel.finish()


def test_an_unknown_total_shows_a_busy_bar_and_no_percentage(window):
    panel = window.progress_panel
    panel.start("Preparing SAM2", total=0, detail="loading the model…")
    panel.advance(3)
    assert panel._count == ""
    panel.finish()


def test_the_total_can_be_revised_mid_job(window):
    panel = window.progress_panel
    panel.start("Export", total=0)
    panel.advance(5, total=10)
    assert panel._count == "50%"
    panel.finish()


def test_finishing_hides_it_again(window):
    panel = window.progress_panel
    panel.start("Export", total=4)
    panel.finish()
    assert panel.isHidden()


def test_stop_is_reported_and_the_button_goes_dead(window):
    panel = window.progress_panel
    stopped = []
    panel.stopRequested.connect(lambda: stopped.append(True))
    panel.start("Segmenting", total=10)
    panel._on_stop()
    assert panel.cancelled and stopped == [True]
    assert not panel.stop_btn.isEnabled(), "one press is enough"
    assert "stopping" in panel.status.toolTip()
    panel.finish()


def test_starting_a_new_job_clears_a_previous_stop(window):
    panel = window.progress_panel
    panel.start("A", total=4)
    panel._on_stop()
    panel.start("B", total=4)
    assert not panel.cancelled and panel.stop_btn.isEnabled()
    panel.finish()


def test_the_text_is_elided_rather_than_widening_the_column(window):
    """The control column is 196 px, and every pixel it takes comes off the
    canvas beside it."""
    panel = window.progress_panel
    panel.start("Segmenting", total=10, detail="a very long detail " * 5)
    panel.advance(5)
    assert len(panel.status.text()) < len(panel.status.toolTip()), \
        "the full text lives in the tooltip"
    panel.finish()


def test_it_draws_no_square_qt_frame_of_its_own(window):
    """The rounded border comes from style.qss, so the corners match the group
    boxes around it."""
    assert window.progress_panel.frameShape() == QFrame.Shape.NoFrame


def test_the_stylesheet_rounds_the_panel_and_the_bar():
    from wormscope import theme
    qss = theme.stylesheet()
    assert "QFrame#progressPanel" in qss
    assert "border-radius" in qss.split("QFrame#progressPanel")[1][:200]
    assert "QProgressBar::chunk" in qss


# --- the main-thread helper -------------------------------------------------
def test_the_progress_helper_drives_the_same_panel(window):
    """Export and pre-encode run on the main thread and use this wrapper; SAM2
    runs on a worker and drives the panel directly. Same panel either way."""
    with Progress(window, "Export overlay", 4, label="Rendering…") as bar:
        assert not window.progress_panel.isHidden()
        assert bar.step(2) is False, "returns True only once Stop was pressed"
        assert window.progress_panel._count == "50%"
    assert window.progress_panel.isHidden()


def test_the_helper_reports_a_stop(window):
    with Progress(window, "Export overlay", 4) as bar:
        window.progress_panel._on_stop()
        assert bar.step(3) is True, "the caller checks this and gives up early"


def test_a_panel_with_no_window_behind_it_still_works():
    """Progress falls back to doing nothing when there is no panel, so batch
    code can call the same helpers."""
    class NoPanel:
        pass
    bar = Progress(NoPanel(), "x", 4)
    assert bar.step(1) is False and bar.cancelled is False
    bar.close()


def test_the_panel_can_be_built_on_its_own(qapp):
    """It is a plain QFrame — nothing in it reaches back into the window."""
    panel = ProgressPanel()
    panel.start("standalone", total=2)
    panel.advance(1)
    assert panel._count == "50%"
    panel.finish()
