"""Upload video — choose which frames to extract before decoding.

Shown when a video file is uploaded (Upload video or image…). Three choices:

    All frames      every frame, at the video's own frame rate
    Frame rate      a lower rate: every n-th frame, so the kept frames stay
                    evenly spaced in time (the rate is rounded to the nearest
                    fps / n, and the dialog says which rate you get)
    Random frames   a number of frames picked at random from the whole video,
                    in time order — e.g. for a training / annotation set. The
                    seed makes the pick reproducible. Random frames are
                    unevenly spaced, so they get no frame rate (speed and
                    acceleration are then per frame and not meaningful).

Only reads the file header (cheap). The choice is returned by
:meth:`sampling` (see io/video.py) and remembered for the next upload.
"""

from pathlib import Path

from PyQt6.QtWidgets import (QButtonGroup, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QGridLayout, QLabel, QRadioButton,
                             QSpinBox, QVBoxLayout)

from .. import settings
from ..io.video import normalize_sampling

MAX_RANDOM = 1_000_000


def _ordinal(n):
    return {1: "every", 2: "every 2nd", 3: "every 3rd"}.get(n, f"every {n}th")


class VideoImportDialog(QDialog):

    def __init__(self, parent, video, info):
        super().__init__(parent)
        self.setWindowTitle("Upload video")
        self.video = Path(video)
        self.fps = info.get("fps")
        self.n = info.get("frames") or 0
        self._build_ui(info)
        self._restore(settings.get("video_sampling") or {})
        self._update()

    # --- widgets -------------------------------------------------------------
    def _build_ui(self, info):
        lay = QVBoxLayout(self)
        facts = []
        if info.get("size"):
            facts.append("{}×{} px".format(*info["size"]))
        facts.append(f"{self.fps:g} fps" if self.fps else "frame rate unknown")
        if self.n:
            facts.append(f"≈ {self.n:,} frames")
        if info.get("duration"):
            facts.append(f"{info['duration']:.1f} s")
        head = QLabel(f"<b>{self.video.name}</b><br>{' · '.join(facts)}")
        lay.addWidget(head)
        lay.addWidget(QLabel("Which frames should be extracted?"))

        grid = QGridLayout()
        self.group = QButtonGroup(self)
        self.all_rb = QRadioButton("All frames")
        self.rate_rb = QRadioButton("Frame rate" if self.fps
                                    else "Every n-th frame")
        self.random_rb = QRadioButton("Random frames")
        for i, rb in enumerate((self.all_rb, self.rate_rb, self.random_rb)):
            self.group.addButton(rb, i)
            rb.toggled.connect(self._update)

        # frame rate (or every n-th frame when the rate is unknown)
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setDecimals(2)
        self.rate_spin.setRange(0.01, self.fps or 1.0)
        self.rate_spin.setSuffix(" fps")
        self.rate_spin.setValue(min(self.fps or 1.0, 5.0))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(2, max(2, self.n or 100_000))
        self.step_spin.setPrefix("every ")
        self.step_spin.setValue(2)
        self.rate_spin.setVisible(bool(self.fps))
        self.step_spin.setVisible(not self.fps)
        self.rate_spin.valueChanged.connect(self._update)
        self.step_spin.valueChanged.connect(self._update)

        # random frames
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, max(1, self.n) if self.n else MAX_RANDOM)
        self.count_spin.setSuffix(" frames")
        self.count_spin.setValue(min(100, self.n) if self.n else 100)
        self.count_spin.valueChanged.connect(self._update)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setPrefix("seed ")
        self.seed_spin.setToolTip("The same seed picks the same frames again")

        self.all_detail = QLabel()
        self.rate_detail = QLabel()
        self.random_detail = QLabel()
        for d in (self.all_detail, self.rate_detail, self.random_detail):
            d.setStyleSheet("color: #5a6b85;")
            d.setWordWrap(True)

        grid.addWidget(self.all_rb, 0, 0)
        grid.addWidget(self.all_detail, 1, 0, 1, 3)
        grid.addWidget(self.rate_rb, 2, 0)
        grid.addWidget(self.rate_spin if self.fps else self.step_spin, 2, 1)
        grid.addWidget(self.rate_detail, 3, 0, 1, 3)
        grid.addWidget(self.random_rb, 4, 0)
        grid.addWidget(self.count_spin, 4, 1)
        grid.addWidget(self.seed_spin, 4, 2)
        grid.addWidget(self.random_detail, 5, 0, 1, 3)
        grid.setColumnStretch(0, 1)
        lay.addLayout(grid)

        note = QLabel(f"Saved as PNG in {self.video.stem}_wormscope/frames/. "
                      "File names keep the video's frame numbers, so frame "
                      "numbers in every CSV point back to the video.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #5a6b85;")
        lay.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Extract frames")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.setMinimumWidth(460)

    def _restore(self, last):
        s = normalize_sampling(last)
        if s["mode"] == "every":
            self.rate_rb.setChecked(True)
            if self.fps:
                self.rate_spin.setValue(self.fps / s["step"])
            else:
                self.step_spin.setValue(s["step"])
        elif s["mode"] == "random":
            self.random_rb.setChecked(True)
            self.count_spin.setValue(s["count"])
            self.seed_spin.setValue(s["seed"])
        else:
            self.all_rb.setChecked(True)

    # --- the numbers ---------------------------------------------------------
    def step(self):
        """Keep every step-th frame for the chosen rate (≥ 1)."""
        if not self.fps:
            return self.step_spin.value()
        return max(1, round(self.fps / self.rate_spin.value()))

    def _update(self, *_):
        self.rate_spin.setEnabled(self.rate_rb.isChecked())
        self.step_spin.setEnabled(self.rate_rb.isChecked())
        self.count_spin.setEnabled(self.random_rb.isChecked())
        self.seed_spin.setEnabled(self.random_rb.isChecked())

        self.all_detail.setText(
            (f"≈ {self.n:,} frames" if self.n else "every frame")
            + (f" at {self.fps:g} fps" if self.fps else ""))

        step = self.step()
        approx = f" · ≈ {-(-self.n // step):,} frames" if self.n else ""
        if self.fps:
            got = self.fps / step
            exact = abs(got - self.rate_spin.value()) < 1e-6
            self.rate_detail.setText(
                f"{_ordinal(step)} frame → {got:g} fps{approx}"
                + ("" if exact else f"  (nearest rate to "
                   f"{self.rate_spin.value():g} that keeps the frames evenly "
                   f"spaced)"))
        else:
            self.rate_detail.setText(f"{_ordinal(step)} frame{approx}")

        self.random_detail.setText(
            f"{self.count_spin.value():,} frames picked at random from the "
            f"whole video, kept in time order. No frame rate: speed and "
            f"acceleration are per frame.")

    def sampling(self) -> dict:
        if self.rate_rb.isChecked():
            step = self.step()
            if step > 1:
                return {"mode": "every", "step": step}
        if self.random_rb.isChecked():
            return {"mode": "random", "count": self.count_spin.value(),
                    "seed": self.seed_spin.value()}
        return {"mode": "all"}

    def accept(self):
        settings.save({"video_sampling": self.sampling()})
        super().accept()
