"""Progress feedback that doesn't take the window hostage.

A small panel at the bottom of the Zoom panel's control column, not a modal
dialog: long jobs (tracking, exporting, pre-encoding) report there — right under
the buttons that started them — while the rest of the GUI stays usable. It
carries a Stop button and hides itself when idle.

Two ways to drive it:

* :class:`Progress` — for work that runs on the main thread (export,
  pre-encode). Same context-manager API as before: ``step()`` returns True when
  the user pressed Stop.
* the widget's own :meth:`ProgressPanel.start` / :meth:`ProgressPanel.advance` /
  :meth:`ProgressPanel.finish` — for work running in a worker thread, where the
  main thread only receives progress signals.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QSizePolicy,
                             QVBoxLayout)

class ProgressPanel(QFrame):
    """The panel itself. Hidden unless something is running."""

    stopRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("progressPanel")
        # No Qt frame: the rounded border comes from style.qss (QFrame
        # #progressPanel), so the corners match the group boxes around it.
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Fixed)
        self._cancelled = False

        # Two lines only: the column is 236 px wide and every pixel it takes
        # comes off the canvas next to it.
        self._title = ""
        self._detail = ""
        self._count = ""
        self.status = QLabel("")
        self.status.setStyleSheet("font-size: 11px;")
        # Ignored width: the text must never widen the control column — it is
        # elided to whatever room there is instead.
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored,
                                  QSizePolicy.Policy.Preferred)
        self.status.setMinimumWidth(0)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("compact")
        self.stop_btn.setFixedSize(50, 18)   # enough for "Stop" plus its padding
        self.stop_btn.clicked.connect(self._on_stop)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        top.addWidget(self.status, 1)
        top.addWidget(self.stop_btn)

        box = QVBoxLayout(self)
        box.setContentsMargins(6, 4, 6, 4)
        box.setSpacing(3)
        box.addLayout(top)
        box.addWidget(self.bar)
        self.hide()

    def _refresh_status(self):
        """One elided line, most useful part first: "Segmenting 48% — worm 1".

        A percentage rather than a count: the frame total is an estimate (a run
        can stop early, and a worm's prompts may not span the whole clip), so
        "47/98" could read past its own total. A percentage of the estimate is
        honest about being approximate. The full text is in the tooltip."""
        text = " ".join(p for p in (self._title, self._count) if p)
        if self._detail:
            text = f"{text} — {self._detail}" if text else self._detail
        metrics = self.status.fontMetrics()
        room = max(40, (self.width() or 190) - 62)   # minus margins + Stop
        self.status.setText(metrics.elidedText(text, Qt.TextElideMode.ElideRight,
                                               room))
        self.status.setToolTip(text)

    # --- driving it ----------------------------------------------------------
    def start(self, title, total=0, detail=""):
        self._cancelled = False
        self._title, self._detail = title, detail
        self._count = ""
        self._refresh_status()
        self.bar.setRange(0, max(0, total))      # 0 total -> busy indicator
        self.bar.setValue(0)
        self.stop_btn.setEnabled(True)
        self.show()

    def advance(self, done, detail=None, total=None):
        if total is not None:
            self.bar.setRange(0, max(0, total))
        if detail is not None:
            self._detail = detail
        top = self.bar.maximum()
        self.bar.setValue(min(done, top) if top else done)
        pct = f"{min(100, round(100 * done / top))}%" if top else ""
        if pct != self._count:
            self._count = pct
            self._refresh_status()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh_status()               # re-elide for the new width

    def finish(self):
        self.hide()

    @property
    def cancelled(self):
        return self._cancelled

    def _on_stop(self):
        self._cancelled = True
        self.stop_btn.setEnabled(False)
        self._detail = "stopping…"
        self._refresh_status()
        self.stopRequested.emit()


class Progress:
    """Main-thread helper with a context-manager API, drawn in the panel
    instead of a modal dialog."""

    def __init__(self, parent, title, total=0, label=None, cancel="Stop"):
        self.total = total
        self.panel = getattr(parent, "progress_panel", None)
        if self.panel is not None:
            self.panel.start(title, total, label or "")
        QApplication.processEvents()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def step(self, done, label=None):
        """Advance to `done`. Returns True if the user asked to stop."""
        if self.panel is None:
            return False
        self.panel.advance(done, label)
        QApplication.processEvents()             # keep the UI alive and clickable
        return self.panel.cancelled

    def set_total(self, total):
        self.total = total
        if self.panel is not None:
            self.panel.advance(self.panel.bar.value(), total=total)

    @property
    def cancelled(self):
        return self.panel is not None and self.panel.cancelled

    def close(self):
        if self.panel is not None:
            self.panel.finish()
