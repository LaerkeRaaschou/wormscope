"""Units… — calibrate µm per pixel from a scale image, and set the frame rate.

Load an image of a scale (a stage micrometer, or any frame with a scale bar),
drag a line along a stretch of known length, and type that length:

    µm per pixel = known length / line length in pixels

The result can also be typed directly. The dialog only *produces* the factor;
every measurement stays in pixels and is converted on display and on Save
(``core.metrics.unit_scales``), so recalibrating never touches the data.

The scale image must have the same pixel size as the video (same camera,
magnification and binning); the dialog warns when the image dimensions differ.
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QVBoxLayout)

from ..config import ZOOM_STEP
from ..core.metrics import UNIT_TO_UM, um_per_px
from ..io.loaders import load_image_rgb
from ..widgets import ScaleView, np_to_qimage

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"


class CalibrationDialog(QDialog):
    """`state` is what a previous run returned (see :meth:`state`), so the
    image, the line and the known length are still there when reopened."""

    def __init__(self, parent, px_to_um=None, fps=None, state=None,
                 frame=None, frame_shape=None, start_dir=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self._frame = frame                # current video frame (array) or None
        self._frame_shape = frame_shape    # (h, w) of the video, for the warning
        self._start_dir = start_dir or str(Path.home())
        self._image_path = None
        self._image_shape = None
        self._build_ui()

        state = state or {}
        self.length_spin.setValue(state.get("length", 100.0))
        self.unit_combo.setCurrentText(state.get("unit", "µm"))
        self.um_spin.setValue(px_to_um or 0.0)
        self.fps_spin.setValue(fps or 0.0)
        if state.get("image") and Path(state["image"]).is_file():
            self._load(state["image"], quiet=True)
        elif state.get("source") == "frame" and frame is not None:
            self._use_frame()
        if state.get("line") and self.view.has_image():
            self.view.set_line(state["line"])
        self._update_status()

    # --- view: zoom and pan ------------------------------------------------------
    def _zoom_step(self, direction):
        self.view.zoom_by(ZOOM_STEP ** direction)

    def _set_pan(self, on):
        self.view.pan_mode = on
        self.view.setCursor(Qt.CursorShape.OpenHandCursor if on
                            else Qt.CursorShape.CrossCursor)

    # --- widgets -------------------------------------------------------------
    def _build_ui(self):
        self.view = ScaleView()
        self.view.clear_image("Load an image of the scale, or use the "
                              "current frame.")
        self.view.lineChanged.connect(self._on_line)

        load_btn = QPushButton("Load scale image…")
        load_btn.clicked.connect(self._choose_image)
        frame_btn = QPushButton("Use current frame")
        frame_btn.setToolTip("Draw on the video itself, e.g. when it has a "
                             "scale bar burnt in")
        frame_btn.setEnabled(self._frame is not None)
        frame_btn.clicked.connect(self._use_frame)
        top = QHBoxLayout()
        top.addWidget(load_btn)
        top.addWidget(frame_btn)
        top.addStretch(1)
        # View tools under the image: Line / Pan, and zoom − 100 % + Fit
        # (the − / + keys too). Zooming in on the bar's ticks makes the line
        # far more precise than drawing it on the whole image.
        self.line_btn = QPushButton("Line")
        self.line_btn.setToolTip("Drag along the scale to draw the line")
        self.pan_btn = QPushButton("Pan")
        self.pan_btn.setToolTip("Drag to move the image (right-drag pans "
                                "with any tool)")
        tools = QButtonGroup(self)
        for b in (self.line_btn, self.pan_btn):
            b.setCheckable(True)
            tools.addButton(b)
        self.line_btn.setChecked(True)
        self.pan_btn.toggled.connect(self._set_pan)
        zoom_out = QPushButton("−")
        zoom_out.setToolTip("Zoom out (−)")
        zoom_out.clicked.connect(lambda: self._zoom_step(-1))
        zoom_in = QPushButton("+")
        zoom_in.setToolTip("Zoom in (+)")
        zoom_in.clicked.connect(lambda: self._zoom_step(1))
        for b in (zoom_out, zoom_in):
            b.setFixedWidth(32)
        self.zoom_label = QLabel("—")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setMinimumWidth(56)
        fit_btn = QPushButton("Fit")
        fit_btn.setToolTip("Show the whole image (or double-click it)")
        fit_btn.clicked.connect(self.view.fit)
        self.view.zoomChanged.connect(
            lambda m: self.zoom_label.setText(f"{m * 100:.0f} %"))
        for key, d in (("+", 1), ("=", 1), ("-", -1)):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda d=d: self._zoom_step(d))
        view_row = QHBoxLayout()
        view_row.addWidget(self.line_btn)
        view_row.addWidget(self.pan_btn)
        view_row.addStretch(1)
        view_row.addWidget(QLabel("Zoom"))
        view_row.addWidget(zoom_out)
        view_row.addWidget(self.zoom_label)
        view_row.addWidget(zoom_in)
        view_row.addWidget(fit_btn)

        hint = QLabel("Drag along the scale. Drag an end to adjust it; hold "
                      "Shift to keep the line straight. Wheel or − / + zooms, "
                      "Pan or right-drag moves the image, double-click fits.")
        hint.setWordWrap(True)
        left = QVBoxLayout()
        left.addLayout(top)
        left.addWidget(self.view, 1)
        left.addLayout(view_row)
        left.addWidget(hint)

        # scale
        self.line_label = QLabel("—")
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.001, 1e6)
        self.length_spin.setDecimals(3)
        self.length_spin.valueChanged.connect(self._recompute)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(list(UNIT_TO_UM))
        self.unit_combo.currentTextChanged.connect(self._recompute)
        known = QHBoxLayout()
        known.addWidget(self.length_spin, 1)
        known.addWidget(self.unit_combo)
        self.um_spin = QDoubleSpinBox()
        self.um_spin.setRange(0.0, 1e6)
        self.um_spin.setDecimals(6)
        self.um_spin.setToolTip("Filled in from the line — or type it")
        self.um_spin.valueChanged.connect(self._update_status)
        scale_form = QFormLayout()
        scale_form.addRow("Line length", self.line_label)
        scale_form.addRow("Known length", known)
        scale_form.addRow("µm per pixel", self.um_spin)
        scale_group = QGroupBox("Scale")
        scale_group.setLayout(scale_form)

        # time
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.0, 1e6)
        self.fps_spin.setDecimals(3)
        self.fps_spin.setSpecialValueText("per frame")   # shown for 0
        time_form = QFormLayout()
        time_form.addRow("Frames per second", self.fps_spin)
        time_group = QGroupBox("Time")
        time_group.setLayout(time_form)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignTop)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        reset = buttons.addButton("Use pixels",
                                  QDialogButtonBox.ButtonRole.ResetRole)
        reset.setToolTip("Forget the calibration: show everything in pixels")
        reset.clicked.connect(lambda: self.um_spin.setValue(0.0))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        right = QVBoxLayout()
        right.addWidget(scale_group)
        right.addWidget(time_group)
        right.addWidget(self.status, 1)
        right.addWidget(buttons)

        root = QHBoxLayout(self)
        root.addLayout(left, 1)
        side = QVBoxLayout()
        side.addLayout(right)
        root.addLayout(side)
        self.resize(1000, 620)

    # --- the image -----------------------------------------------------------
    def _choose_image(self):
        start = str(Path(self._image_path).parent) if self._image_path \
            else self._start_dir
        path, _ = QFileDialog.getOpenFileName(self, "Scale image", start,
                                              IMAGE_FILTER)
        if path:
            self._load(path)

    def _load(self, path, quiet=False):
        try:
            arr = load_image_rgb(path)
        except Exception as err:             # noqa: BLE001 - shown to the user
            if not quiet:
                QMessageBox.warning(self, "Scale image",
                                    f"Could not open\n{path}\n\n{err}")
            return
        self._image_path = str(path)
        self._show(arr)

    def _use_frame(self):
        if self._frame is not None:
            self._image_path = None
            self._show(self._frame)

    def _show(self, arr):
        self._image_shape = arr.shape[:2]
        self.view.set_image(np_to_qimage(arr), arr.shape[1], arr.shape[0])
        self.view.set_line(None)
        self._update_status()

    # --- the numbers ---------------------------------------------------------
    def _on_line(self, length_px):
        self.line_label.setText(f"{length_px:.1f} px" if length_px else "—")
        self._recompute()

    def _recompute(self):
        """Line or known length changed: derive µm per pixel from them."""
        value = um_per_px(self.view.length(), self.length_spin.value(),
                          self.unit_combo.currentText())
        if value is not None:
            self.um_spin.setValue(value)
        self._update_status()

    def _update_status(self):
        lines = []
        um = self.um_spin.value()
        if um:
            lines.append(f"1 px = {um:.4g} µm\n1 µm = {1 / um:.4g} px")
        else:
            lines.append("Not calibrated: metrics are shown in pixels.")
        if (self._image_shape and self._frame_shape
                and tuple(self._image_shape) != tuple(self._frame_shape)):
            ih, iw = self._image_shape
            vh, vw = self._frame_shape
            lines.append(f"\n⚠ The scale image is {iw}×{ih} px but the video "
                         f"is {vw}×{vh} px. The calibration is only valid if "
                         f"both have the same pixel size (camera, "
                         f"magnification, binning).")
        self.status.setText("\n".join(lines))

    # --- results -------------------------------------------------------------
    def px_to_um(self):
        return self.um_spin.value() or None

    def fps(self):
        return self.fps_spin.value() or None

    def state(self):
        """What to pass back in next time the dialog opens."""
        return {"image": self._image_path,
                "source": "file" if self._image_path else
                          ("frame" if self.view.has_image() else None),
                "line": self.view.line,
                "length": self.length_spin.value(),
                "unit": self.unit_combo.currentText()}
