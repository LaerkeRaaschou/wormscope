"""Widget construction for the main window — the only file you need to touch to
add a button, a slider or a whole new panel.

Everything created here is stored on ``self`` so the behaviour mixins can reach
it. Signals are wired to methods that live in the other mixins.

Layout::

    +----------+  +--------------------------+  +-----------+
    | Labelled |  |          VIDEO           |  |   PLOTS   |
    | worm list|  |     (frame + masks)      |  |  (4 slots)|
    |  + Info  |  | [<][▶][>] frame ===  [E] |  |           |
    +----------+--+--------------------------+  |           |
    | view  |            ZOOM                |  |           |
    | [AI]  |        (edit here)             |  |           |
    | [Man] |  controls left, canvas right   |  |           |
    +----------------------------------------+  +-----------+

The controls under both images are deliberately compact (CONTROL_H): the frames
are what people look at, so the widgets around them stay out of the way.
"""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QAbstractButton, QButtonGroup, QCheckBox, QFrame,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QListWidget, QPushButton, QScrollArea, QSlider,
                             QVBoxLayout, QWidget)

from ..config import (ANNOTATION_AUTOMATIC, ANNOTATION_MANUAL, BRUSH_MAX,
                      BRUSH_MIN, PLAYBACK_FPS, SMOOTH_WINDOW)
from ..widgets import PlotPanel, VideoView, ZoomView
from .progress import ProgressPanel


CONTROL_H = 24          # height of the small controls around the images
ZOOM_CONTROL_W = 236    # width of the Zoom panel's control column
ROW_SPACING = 4


def _compact(*widgets):
    """Shrink controls so the images get the space. Returns the widgets."""
    for w in widgets:
        w.setObjectName("compact")
        w.setFixedHeight(CONTROL_H)
    return widgets


def _tight(layout):
    """A control row with no wasted padding."""
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(ROW_SPACING)
    return layout


class LayoutMixin:
    """Builds every widget and lays the window out. No behaviour lives here."""

    def _build_ui(self):
        self._build_video_panel()
        self._build_zoom_panel()
        self._build_labelled_panel()
        self._build_plots_panel()
        self._assemble()
        self._relax_focus()
        self._build_shortcuts()

    # --- keyboard ------------------------------------------------------------
    def _relax_focus(self):
        """Buttons and sliders are reachable by Tab, but never *take* focus
        when clicked.

        A control that keeps focus after a click quietly swallows the keyboard:
        Space re-presses the button you last used instead of playing, and ← / →
        nudge the slider you last dragged. Since every one of these is really a
        mouse control, they give focus back to the window and the keys keep
        meaning what they say. The worm list is left alone — Up / Down through
        the worms is worth having."""
        for kind in (QAbstractButton, QSlider):
            for w in self.findChildren(kind):
                w.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def _build_shortcuts(self):
        """Window-wide keys, matched before the focused widget sees them.

        ← / → are pressed constantly and are exactly the keys a focused slider,
        list or plot canvas would otherwise eat, so they are shortcuts rather
        than plain key presses: stepping frames then works no matter where you
        last clicked. Shift steps ten at a time for crossing a long clip."""
        self._shortcuts = []
        for key, delta in (("Left", -1), ("Right", 1),
                           ("Shift+Left", -10), ("Shift+Right", 10)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(lambda d=delta: self._step(d))
            self._shortcuts.append(sc)
        # + / − zoom the editing canvas. "=" is the unshifted + key on US
        # layouts, and the keypad keys arrive as the same sequences.
        for key, d in (("+", 1), ("=", 1), ("-", -1)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(lambda d=d: self._zoom_step(d))
            self._shortcuts.append(sc)
        # Pan the editing canvas: W A S D, or Alt (Option on a Mac) + arrows —
        # the plain arrows already step frames.
        for keys, (dx, dy) in ((("W", "Alt+Up"), (0, -1)),
                               (("S", "Alt+Down"), (0, 1)),
                               (("A", "Alt+Left"), (-1, 0)),
                               (("D", "Alt+Right"), (1, 0))):
            for key in keys:
                sc = QShortcut(QKeySequence(key), self)
                sc.setContext(Qt.ShortcutContext.WindowShortcut)
                sc.activated.connect(lambda x=dx, y=dy: self._pan_step(x, y))
                self._shortcuts.append(sc)
        # Undo / redo / save. Qt's "Ctrl" is ⌘ Command on a Mac and "Meta" is the
        # Control key there, so listing both makes ⌘Z and Ctrl+Z work on
        # every platform. As shortcuts they also work while a slider or the
        # worm list has focus.
        for key, slot in (("Ctrl+Z", self._undo_action),
                          ("Meta+Z", self._undo_action),
                          ("Ctrl+Shift+Z", self._redo_action),
                          ("Meta+Shift+Z", self._redo_action),
                          ("Ctrl+Y", self._redo_action),
                          # Save: ⌘S / Ctrl+S; Save as…: ⌘⇧S / Ctrl+Shift+S
                          ("Ctrl+S", self._save),
                          ("Meta+S", self._save),
                          ("Ctrl+Shift+S", self._save_as),
                          ("Meta+Shift+S", self._save_as)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

    # --- Video panel ---------------------------------------------------------
    def _build_video_panel(self):
        self.video = VideoView()
        self.video.imageClicked.connect(self._on_video_click)

        # One compact control strip under the image: transport, frame counter,
        # scrubber and export. Everything is small on purpose — the frame itself
        # is what deserves the space.
        self.prev_btn = prev_btn = QPushButton("<")
        self.next_btn = next_btn = QPushButton(">")
        self.play_btn = QPushButton("▶")
        for b in (prev_btn, self.play_btn, next_btn):
            b.setObjectName("compact")
            b.setFixedSize(30, CONTROL_H)
        prev_btn.setToolTip("Previous frame (←)")
        next_btn.setToolTip("Next frame (→)")
        self.play_btn.setToolTip("Play / pause (Space)")
        prev_btn.clicked.connect(lambda: self._step(-1))
        next_btn.clicked.connect(lambda: self._step(1))
        self.play_btn.clicked.connect(self._toggle_play)
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setFixedWidth(96)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, self.n - 1))
        self.slider.valueChanged.connect(self.set_frame)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._interval = max(1, int(1000 / PLAYBACK_FPS))

        self.export_btn = QPushButton("Export…")
        self.export_btn.setObjectName("compact")
        self.export_btn.setFixedHeight(CONTROL_H)
        self.export_btn.setToolTip("Render every frame with your edited masks + "
                                   "skeletons to an .mp4")
        self.export_btn.clicked.connect(self._export_video)

        nav = QHBoxLayout()
        nav.setContentsMargins(0, 0, 0, 0)
        nav.setSpacing(4)
        nav.addWidget(prev_btn)
        nav.addWidget(self.play_btn)
        nav.addWidget(next_btn)
        nav.addWidget(self.frame_label)
        nav.addWidget(self.slider, 1)
        nav.addWidget(self.export_btn)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)
        vbox.addWidget(self.video, 1)
        vbox.addLayout(nav)
        self.video_group = QGroupBox("Video")
        self.video_group.setLayout(vbox)

    # --- Zoom panel (editing canvas) ----------------------------------------
    def _build_zoom_panel(self):
        self.progress_panel = ProgressPanel(self)
        self.progress_panel.stopRequested.connect(self._on_progress_stop)
        self.zoomv = ZoomView()
        self.zoomv.paintRequested.connect(self._on_paint)
        self.zoomv.strokeFinished.connect(self._on_stroke_finished)
        self.zoomv.viewportChanged.connect(self._on_viewport_changed)
        self.zoomv.pointAdded.connect(self._on_sam_point)
        self.zoomv.measureAdded.connect(self._on_measure_added)
        self.zoomv.measureMoved.connect(self._on_measure_moved)
        self.zoomv.measureRemoved.connect(self._on_measure_removed)
        self.zoomv.measureMoveFinished.connect(self._on_measure_move_finished)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.btn_pan = QPushButton("Pan")
        self.btn_draw = QPushButton("Draw")
        self.btn_draw.setToolTip("Paint mask.\nHold Shift to erase for one "
                                 "stroke without switching tools.")
        self.btn_erase = QPushButton("Erase")
        self.btn_erase.setToolTip("Rub mask out.\nHold Shift to paint for one "
                                  "stroke without switching tools.")
        # Two sticky SAM tools; Shift inverts whichever one is active for a
        # single click, so "+ with an occasional Shift" and "− permanently"
        # are both one gesture away.
        self.btn_sam_pos = QPushButton("+")
        self.btn_sam_pos.setToolTip("Points are the worm.\n"
                                    "Shift+click for a single NOT-the-worm point.")
        self.btn_sam_neg = QPushButton("−")
        self.btn_sam_neg.setToolTip("Points are NOT the worm — stays negative "
                                    "until you switch back.\n"
                                    "Shift+click for a single worm point.")
        self.btn_sam = self.btn_sam_pos          # the default SAM tool
        # Manual length: click points along the selected worm.
        self.btn_measure = QPushButton("Measure")
        self.btn_measure.setToolTip(
            "Measure the selected worm's length by hand (L): click points "
            "along it, head to tail.\nDrag a point to move it, click on the "
            "line to insert one,\nShift+click a point or Backspace to remove "
            "one. Right-drag pans.")
        self.measure_clear_btn = QPushButton("Clear")
        self.measure_clear_btn.setToolTip("Remove the selected worm's manual "
                                          "line on this frame")
        self.measure_clear_btn.clicked.connect(self._measure_clear)
        self.measure_label = QLabel("—")
        # Annotation mode: which data (and tools) the window shows.
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_auto_btn = QPushButton("Automatic")
        self.mode_auto_btn.setToolTip(
            "Segmentation: masks, SAM2 and brush tools, area / length / speed "
            "plots.\nManual length lines are hidden (and kept).")
        self.mode_manual_btn = QPushButton("Manual")
        self.mode_manual_btn.setToolTip(
            "Manual length: click points along the worm.\nMasks and their "
            "metrics are hidden (and kept). The worm list shows every worm.")
        for b, m in ((self.mode_auto_btn, ANNOTATION_AUTOMATIC),
                     (self.mode_manual_btn, ANNOTATION_MANUAL)):
            b.setCheckable(True)
            b.clicked.connect(lambda _c, m=m: self._set_annotation_mode(m))
            self.mode_group.addButton(b)
        self.measure_label.setToolTip("Manual length of the selected worm on "
                                      "this frame")
        for b, mode in ((self.btn_pan, ZoomView.MODE_PAN),
                        (self.btn_draw, ZoomView.MODE_DRAW),
                        (self.btn_erase, ZoomView.MODE_ERASE),
                        (self.btn_sam_pos, ZoomView.MODE_SAM_POS),
                        (self.btn_sam_neg, ZoomView.MODE_SAM_NEG),
                        (self.btn_measure, ZoomView.MODE_MEASURE)):
            b.setCheckable(True)
            b.clicked.connect(lambda _c, m=mode: self._set_mode(m))
            self.tool_group.addButton(b)
        self.btn_pan.setChecked(True)

        brush_lbl = QLabel("Brush")
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(BRUSH_MIN, BRUSH_MAX)
        self.brush_slider.setValue(self.zoomv.brush)
        self.brush_slider.valueChanged.connect(self._set_brush)
        self.brush_val = QLabel(str(self.zoomv.brush))

        self.outline_btn = QPushButton("Outline")
        self.outline_btn.setCheckable(True)
        self.outline_btn.setToolTip("Show masks as an outline instead of a "
                                    "filled overlay, so you can see the worm")
        self.outline_btn.toggled.connect(self._toggle_outline)

        self.sam_run_btn = QPushButton("Segment ▶ all")
        self.sam_run_btn.setToolTip("Propagate the point prompts through every "
                                    "frame with SAM2")
        self.sam_run_btn.clicked.connect(lambda: self._sam_propagate())
        self.sam_mask_btn = QPushButton("Mask → prompt")
        self.sam_mask_btn.setToolTip(
            "Hand this worm's current mask to SAM2 as the prompt for this "
            "frame — fix a frame with the brush, then re-segment from here")
        self.sam_mask_btn.clicked.connect(self._sam_mask_prompt)
        self.sam_again_btn = QPushButton("Re-segment ▶ here")
        self.sam_again_btn.setToolTip(
            "Track again from the current frame onwards, using the corrections "
            "you just made on it")
        self.sam_again_btn.clicked.connect(self._sam_resegment)
        self.back_box = QCheckBox("back")
        self.back_box.setToolTip("Also re-track from this frame back towards frame 0")
        # Belongs with the brush, not with SAM2: it is about what happens to
        # the frames *you* painted.
        self.lock_box = QCheckBox("lock edits")
        self.lock_box.setToolTip(
            "Frames you corrected by hand keep your mask: SAM2 still uses them "
            "as prompts to track from, but never writes over them.")
        self.lock_box.toggled.connect(self._settings_changed)
        self.sam_clear_btn = QPushButton("Clear pts")
        self.sam_clear_btn.setToolTip("Forget the SAM2 point prompts")
        self.sam_clear_btn.clicked.connect(self._sam_clear)

        # Zoom as a percentage with − / + steps (keys − and + do the same).
        # 100 % = one image pixel per screen pixel.
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setToolTip("Zoom out (−)")
        self.zoom_out_btn.clicked.connect(lambda: self._zoom_step(-1))
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setToolTip("Zoom in (+)")
        self.zoom_in_btn.clicked.connect(lambda: self._zoom_step(1))
        self.zoom_label = QLabel(self._zoom_text())
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setMinimumWidth(48)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip("Frame the selected worm (whole frame if none)")
        self.fit_btn.clicked.connect(self._fit_view)
        self.zoom_full_btn = QPushButton("Zoom out")
        self.zoom_full_btn.setToolTip("Show the whole frame")
        self.zoom_full_btn.clicked.connect(self._zoom_out_full)
        self.undo_btn = QPushButton("↶")
        self.undo_btn.setFixedWidth(34)
        self.undo_btn.setToolTip("Undo (⌘Z / Ctrl+Z)")
        self.undo_btn.clicked.connect(self._undo_action)
        self.undo_btn.setEnabled(False)
        self.redo_btn = QPushButton("↷")
        self.redo_btn.setFixedWidth(34)
        self.redo_btn.setToolTip("Redo (⌘⇧Z / Ctrl+Shift+Z)")
        self.redo_btn.clicked.connect(self._redo_action)
        self.redo_btn.setEnabled(False)
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Save (⌘S / Ctrl+S)")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        self.save_as_btn = QPushButton("Save as…")
        self.save_as_btn.setToolTip("Save the whole video to another folder "
                                    "and keep working there (⌘⇧S / "
                                    "Ctrl+Shift+S)")
        self.save_as_btn.clicked.connect(self._save_as)
        self.save_folder_label = QLabel()
        self.save_folder_label.setStyleSheet("color: #5a6b85; font-size: 11px;")
        self.save_folder_label.setWordWrap(True)

        # Every control is shrunk to CONTROL_H; sliders lose their fixed widths
        # so they can breathe in the narrow column.
        _compact(self.btn_pan, self.btn_draw, self.btn_erase, self.outline_btn,
                 self.btn_sam_pos, self.btn_sam_neg, self.sam_mask_btn,
                 self.btn_measure, self.measure_clear_btn,
                 self.mode_auto_btn, self.mode_manual_btn,
                 self.sam_clear_btn, self.sam_run_btn, self.sam_again_btn,
                 self.zoom_out_btn, self.zoom_in_btn, self.fit_btn,
                 self.zoom_full_btn, self.undo_btn, self.redo_btn, self.save_btn,
                 self.save_as_btn)
        for b in (self.zoom_out_btn, self.zoom_in_btn):
            b.setFixedWidth(28)
        self.brush_slider.setMinimumWidth(60)

        zbox = QHBoxLayout()                 # controls on the left, image right
        zbox.setContentsMargins(4, 4, 4, 4)
        zbox.setSpacing(ROW_SPACING + 2)
        zbox.addLayout(self._zoom_controls(brush_lbl))
        zbox.addWidget(self.zoomv, 1)
        self.zoom_group = QGroupBox("Zoom  (edit here)")
        self.zoom_group.setLayout(zbox)

    def _tool_box(self, title):
        """A small outlined box with a title, for one family of tools.
        Returns (box, grid) — fill the grid, the box goes in the column."""
        box = QGroupBox(title)
        box.setObjectName("toolbox")
        grid = QGridLayout(box)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(ROW_SPACING)
        grid.setVerticalSpacing(ROW_SPACING - 1)
        return box, grid

    def _zoom_controls(self, brush_lbl):
        """The Zoom panel's controls as a narrow column down the left side.

        Top to bottom: the Automatic | Manual switch (ui/annotation_mode.py,
        it shows one mode's boxes and hides the other's), the view tools
        everybody needs (Pan, Outline), then four
        small boxes — "Zoom", "Manual length" (the Measure tool), "AI edits"
        (SAM2 prompts and tracking) and "Manual edits" (brush and what happens
        to the frames you painted) —
        then undo / redo / save and progress. Kept in a column so the canvas gets the height."""
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(ROW_SPACING + 2)

        # --- annotation mode: Automatic | Manual ---
        mode = _tight(QHBoxLayout())
        mode.addWidget(self.mode_auto_btn, 1)
        mode.addWidget(self.mode_manual_btn, 1)
        col.addLayout(mode)

        # --- view: always visible, at the top ---
        top = _tight(QHBoxLayout())
        top.addWidget(self.btn_pan, 1)
        top.addWidget(self.outline_btn, 1)
        col.addLayout(top)

        # --- Zoom: − % + on one row, Fit and Zoom out beneath ---
        zbx, g = self._tool_box("Zoom")
        zrow = _tight(QHBoxLayout())
        zrow.addWidget(self.zoom_out_btn)
        zrow.addWidget(self.zoom_label, 1)
        zrow.addWidget(self.zoom_in_btn)
        g.addLayout(zrow, 0, 0, 1, 2)
        g.addWidget(self.fit_btn, 1, 0)
        g.addWidget(self.zoom_full_btn, 1, 1)
        col.addWidget(zbx)

        # --- Manual length: the Measure tool, Clear, and the result ---
        self.measure_box = ml = self._tool_box("Manual length")[0]
        g = ml.layout()
        g.addWidget(self.btn_measure, 0, 0)
        g.addWidget(self.measure_clear_btn, 0, 1)
        g.addWidget(self.measure_label, 1, 0, 1, 2)
        col.addWidget(ml)

        # --- AI edits: SAM2 prompts, then tracking ---
        self.ai_box = ai = self._tool_box("AI edits")[0]
        g = ai.layout()
        g.addWidget(self.btn_sam_pos, 0, 0)
        g.addWidget(self.btn_sam_neg, 0, 1)
        g.addWidget(self.sam_clear_btn, 1, 0, 1, 2)
        g.addWidget(self.sam_run_btn, 2, 0, 1, 2)
        again = _tight(QHBoxLayout())        # “back” modifies the button beside it
        again.addWidget(self.sam_again_btn, 1)
        again.addWidget(self.back_box)
        g.addLayout(again, 3, 0, 1, 2)
        col.addWidget(ai)

        # --- Manual edits: Draw / Erase side by side, brush beneath ---
        self.brush_box = man = self._tool_box("Manual edits")[0]
        g = man.layout()
        g.addWidget(self.btn_draw, 0, 0)
        g.addWidget(self.btn_erase, 0, 1)
        brush = _tight(QHBoxLayout())
        brush.addWidget(brush_lbl)
        brush.addWidget(self.brush_slider, 1)
        brush.addWidget(self.brush_val)
        g.addLayout(brush, 1, 0, 1, 2)
        g.addWidget(self.sam_mask_btn, 2, 0, 1, 2)
        g.addWidget(self.lock_box, 3, 0, 1, 2)
        col.addWidget(man)

        # --- history ---
        hist = _tight(QHBoxLayout())
        hist.addWidget(self.undo_btn)
        hist.addWidget(self.redo_btn)
        hist.addWidget(self.save_btn, 1)
        hist.addWidget(self.save_as_btn, 1)
        col.addLayout(hist)
        col.addWidget(self.save_folder_label)   # where Save writes

        # --- progress ---
        # It sits in the column rather than floating: a long job then
        # reports right next to the buttons that started it.
        col.addWidget(self.progress_panel)
        col.addStretch(1)                    # keep everything pinned to the top

        holder = QWidget()
        holder.setLayout(col)

        # Inside a scroll area so a short window (or a platform with larger
        # fonts than this one) scrolls the column instead of forcing the whole
        # window taller or clipping the last controls.
        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedWidth(ZOOM_CONTROL_W)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; } "
                             "QScrollArea > QWidget > QWidget "
                             "{ background: transparent; }")
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return outer

    # --- Labelled worm list --------------------------------------------------
    def _build_labelled_panel(self):
        self.list = QListWidget()
        for w in self.worms:
            self.list.addItem(f"#{w.obj_id}")
        self.list.currentRowChanged.connect(self._on_list_row)
        # Two ways in: start a new video from raw frames, or carry on editing
        # one that was saved before. "New" comes first — it is how every
        # video starts.
        self.import_btn = QPushButton("Upload video or image…")
        self.import_btn.setToolTip("A video file (.mp4 .avi .mov …) is decoded "
                                   "into frames; a folder of raw frames opens "
                                   "as a video; a single .png / .jpg / .tif as "
                                   "a photo")
        self.import_btn.clicked.connect(self._import_raw)
        self.open_btn = QPushButton("Continue editing video…")
        self.open_btn.setToolTip("Open a video you worked on before (a folder "
                                 "containing frames/ and binary/)")
        self.open_btn.clicked.connect(self._open_folder)

        self.add_btn = QPushButton("Add")
        self.add_btn.setToolTip("New empty worm, then click it with + (AI edits) (N)")
        self.add_btn.clicked.connect(self._add_worm)
        self.del_btn = QPushButton("Delete")
        self.del_btn.setToolTip("Delete the selected worm (all frames)")
        self.del_btn.clicked.connect(self._delete_worm)
        self.id_btn = QPushButton("Reassign ID")
        self.id_btn.setToolTip("Reassign the selected worm's ID")
        self.id_btn.clicked.connect(self._reassign_id)
        wrow1 = QHBoxLayout()
        wrow1.setContentsMargins(0, 0, 0, 0)
        wrow1.setSpacing(4)
        wrow1.addWidget(self.add_btn, 1)
        wrow1.addWidget(self.del_btn, 1)
        wbtns = QVBoxLayout()
        wbtns.setContentsMargins(0, 0, 0, 0)
        wbtns.setSpacing(4)
        wbtns.addLayout(wrow1)               # row 1: Add | Delete
        wbtns.addWidget(self.id_btn)         # row 2: Reassign ID (full width)

        self.info = QLabel("Select a worm to inspect or edit it.")
        self.info.setWordWrap(True)
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop)
        lbox = QVBoxLayout()
        lbox.addWidget(self.import_btn)
        lbox.addWidget(self.open_btn)
        lbox.addWidget(self.list, 1)
        lbox.addLayout(wbtns)
        lbox.addWidget(self.info)
        self.labelled_group = QGroupBox("Labelled")
        self.labelled_group.setLayout(lbox)
        self.labelled_group.setMinimumWidth(210)
        self.labelled_group.setMaximumWidth(230)

    # --- Plots ---------------------------------------------------------------
    def _build_plots_panel(self):
        self.plot = PlotPanel(self.plot_metrics)
        # Area and length are drawn relative to their own median: the shape of
        # the curve is what matters, and it stays readable whatever the units.
        # Hovering a curve shows the real value.
        self.plot.set_normalized({self.area_metric, self.length_metric})
        self.plot.set_manual_metric(self.length_metric)   # manual lengths: ◆
        self.plot.setToolTip("Area and length are shown ÷ their median — "
                             "hover a curve for the real value.\n"
                             "Orange diamonds on the length plot are the "
                             "manual lengths.\n"
                             "Click or drag to scrub through frames.")
        self.plot.frameSelected.connect(self.set_frame)
        self.smooth_box = QCheckBox("Smooth")
        self.smooth_box.setToolTip(f"Moving-average smoothing of the curves "
                                   f"(window {SMOOTH_WINDOW}) — display only")
        self.smooth_box.toggled.connect(self._toggle_smooth)
        self.units_btn = QPushButton("Units…")
        self.units_btn.setToolTip("Calibrate: draw a line on a scale image to "
                                  "set µm/pixel, and set fps, to show metrics "
                                  "in physical units")
        self.units_btn.clicked.connect(self._set_units)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Recompute area / length / speed / "
                                    "acceleration for all worms from the masks")
        self.refresh_btn.clicked.connect(self._refresh_plots)
        prow = QHBoxLayout()
        prow.addWidget(self.smooth_box)
        prow.addWidget(self.units_btn)
        prow.addStretch()
        prow.addWidget(self.refresh_btn)
        pbox = QVBoxLayout()
        pbox.addLayout(prow)
        pbox.addWidget(self.plot)
        self.plot_group = QGroupBox("Plots")
        self.plot_group.setLayout(pbox)

    # --- put the four panels in the grid ------------------------------------
    def _on_progress_stop(self):
        if getattr(self, "sam", None) is not None:
            self.sam.stop_current()

    def _assemble(self):
        grid = QGridLayout()
        # The worm list sits beside the overview video only; the Zoom canvas —
        # where the actual editing happens — then spans the full width beneath
        # both of them and gets the larger share of the height.
        grid.addWidget(self.labelled_group, 0, 0)
        grid.addWidget(self.video_group,    0, 1)
        grid.addWidget(self.zoom_group,     1, 0, 1, 2)
        grid.addWidget(self.plot_group,     0, 2, 2, 1)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 4)          # the image panels get the width
        grid.setColumnStretch(2, 2)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 4)             # the editing canvas gets the height
        central = QWidget()
        central.setLayout(grid)
        self.setCentralWidget(central)
        self.resize(1200, 860)
