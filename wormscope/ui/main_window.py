"""The main window: state, frame navigation, selection and playback.

The window is deliberately thin — it owns the state and the "what is selected /
which frame are we on" logic, and inherits the rest from one mixin per area:

    LayoutMixin    wormscope/ui/layout.py      builds every widget
    RenderMixin    wormscope/ui/rendering.py   overlay, skeleton, image adjust
    EditingMixin   wormscope/ui/editing.py     brush tools, metric refresh
    HistoryMixin   wormscope/ui/history.py     undo / redo
    WormListMixin  wormscope/ui/worms.py       add / delete / reassign worms
    SamMixin       wormscope/ui/sam.py         SAM2 point prompts
    IOActionsMixin wormscope/ui/io_actions.py  open, import, units, save, export

To add a feature: put the widget in layout.py and the behaviour in the matching
mixin (or a new one — add it to the bases below).
"""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow

from ..config import (ACCEL_METRIC, HIT_RADIUS, KINEMATIC_METRICS, MAX_PLOTS,
                      SPEED_METRIC)
from ..core.models import Dataset, Worm
from ..core.utils import find_col
from .editing import EditingMixin
from .history import HistoryMixin
from .io_actions import IOActionsMixin
from .layout import LayoutMixin
from .rendering import RenderMixin
from .sam import SamMixin
from .worms import WormListMixin

# Keep references so windows opened from "Open video…" aren't garbage-collected.
WINDOWS: list = []


class MainWindow(LayoutMixin, RenderMixin, EditingMixin, HistoryMixin,
                 WormListMixin, SamMixin, IOActionsMixin, QMainWindow):
    HIT_RADIUS = HIT_RADIUS

    def __init__(self, ds: Dataset):
        super().__init__()
        self._init_state(ds)
        self._build_ui()                     # LayoutMixin
        self._finish_setup()

    # --- construction --------------------------------------------------------
    def _init_state(self, ds: Dataset):
        self.ds = ds
        self.frames = ds.frames
        self.worms = ds.worms
        self.masks = ds.masks
        self.n = len(ds.frames)
        self.frame = 0
        self.selected: Worm | None = None
        self._overlay: np.ndarray | None = None
        self._sel_skel: np.ndarray | None = None
        self._last_paint = None
        self._base_title = ds.title
        self._fitted = False
        self.by_id = {w.obj_id: w for w in self.worms}
        self.onion = False
        self.outline_only = False            # show masks as outlines, not filled
        self.px_to_um = None                 # calibration: µm per pixel
        self.fps = None                      # calibration: frames per second
        self._data_dir = ds.csv_path.parent if ds.csv_path else None
        self._frames_dir = ds.frames_dir     # on-disk frames for SAM2
        self.sam = None                      # SamWorker: SAM2 on its own thread
        self._sam_ready = False              # its backend finished loading
        self._sam_points = {}                # (obj_id, frame) -> [(x, y, label), ...]
        self._hand_edited: set = set()       # (obj_id, frame) painted by hand,
        self._anchored: set = set()          # ...and those already given to SAM2
        self._prompted_worms: set = set()    # worms SAM2 has prompts for
        self._tracked: set = set()           # worms touched by the running job
        self._edited_during_run: set = set()  # edits made while it runs — kept
        self._kept_locked: set = set()       # results dropped by “lock edits”
        self._undo: list = []                # (undo_fn, redo_fn)
        self._redo: list = []
        self._stroke_before = None           # mask snapshot at stroke start
        self._stroke_key = None
        self._pending_delete: set = set()    # files to unlink on Save

        self.area_metric = find_col(ds.metric_names, ["area_px", "area"]) or "area_px"
        self.length_metric = find_col(ds.metric_names, ["length_px", "length"]) or "length_px"

        # 4 plot slots: mask metrics (area, length) + kinematics (speed, accel)
        self.plot_metrics = list(ds.metric_names[:2])
        for p in KINEMATIC_METRICS:
            if len(self.plot_metrics) < MAX_PLOTS and p not in self.plot_metrics:
                self.plot_metrics.append(p)
        for w in self.worms:                 # populate speed / acceleration
            self._recompute_kinematics(w)

    def _finish_setup(self):
        self.list.setCurrentRow(-1)          # start with NO worm selected
        self.plot.set_worm(None)
        self._update_title()
        if self.n:
            self.set_frame(0)
            # Load the model and embed frame 0 in the background: that is most
            # of the cost of the first SAM click, and there is no reason to make
            # the user wait for it while they are still looking at the video.
            self._start_sam(warm=True)
        else:
            self._show_empty_state()

    def _show_empty_state(self):
        """No video loaded: grey out everything that needs frames and say how
        to get one. Only “Open video…” and “Import raw frames…” stay live."""
        for w in (self.prev_btn, self.next_btn, self.play_btn, self.slider,
                  self.export_btn, self.add_btn,
                  self.del_btn, self.id_btn, self.btn_pan, self.btn_draw,
                  self.btn_erase, self.btn_sam_pos, self.btn_sam_neg,
                  self.outline_btn, self.onion_btn, self.prop_btn,
                  self.brush_slider, self.zoom_slider, self.fit_btn,
                  self.sam_run_btn, self.sam_again_btn, self.sam_mask_btn,
                  self.sam_clear_btn, self.back_box, self.lock_box,
                  self.refresh_btn,
                  self.units_btn, self.smooth_box, self.list):
            w.setEnabled(False)
        self.frame_label.setText("no frames")
        self.info.setText("No video loaded.\n\nOpen a folder that contains "
                          "frames/ (and usually binary/), or import a folder "
                          "of raw frames to segment from scratch.")

    # --- frame / selection ---------------------------------------------------
    def set_frame(self, f):
        if not self.n:
            return
        f = max(0, min(self.n - 1, f))
        self.frame = f
        self._refresh_sel_skel()
        self._render_overlay()
        self.video.set_selection(self.selected.centroid(f) if self.selected else None)
        self.plot.set_frame(f)
        self._update_info()
        self.frame_label.setText(f"frame {f} / {self.n - 1}")
        self.slider.blockSignals(True)
        self.slider.setValue(f)
        self.slider.blockSignals(False)

    def select_worm(self, worm):
        self.selected = worm
        row = self.worms.index(worm) if worm is not None else -1
        self.list.blockSignals(True)
        self.list.setCurrentRow(row)
        self.list.blockSignals(False)
        self.plot.set_worm(worm)
        self.plot.set_frame(self.frame)
        self._refresh_sel_skel()
        self._render_overlay()
        self.video.set_selection(worm.centroid(self.frame) if worm else None)
        bbox = worm.bbox(self.frame) if worm else None
        if bbox is not None:
            self.zoomv.frame_bbox(bbox)      # zoom in on the picked worm
        self._update_info()

    def _on_list_row(self, row):
        if 0 <= row < len(self.worms):
            self.select_worm(self.worms[row])

    def _on_video_click(self, x, y):
        xi, yi = int(x), int(y)
        # 1) select any worm whose MASK covers the clicked pixel (whole body is
        #    clickable). If bodies overlap, prefer the smaller one.
        hits = []
        for w in self.worms:
            bbox = w.bbox(self.frame)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            if not (x0 <= xi < x1 and y0 <= yi < y1):
                continue
            m = self.masks.get(w.obj_id, self.frame)
            if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and m[yi, xi]:
                area = w.metrics.get(self.area_metric, {}).get(self.frame) \
                    or (x1 - x0) * (y1 - y0)
                hits.append((area, w))
        if hits:
            hits.sort(key=lambda t: t[0])
            self.select_worm(hits[0][1])
            return
        # 2) fall back to nearest centroid within a small radius (thin masks /
        #    just-missed clicks)
        best, best_d = None, self.HIT_RADIUS
        for w in self.worms:
            c = w.centroid(self.frame)
            if c is None:
                continue
            d = ((c[0] - x) ** 2 + (c[1] - y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = w, d
        if best is not None:
            self.select_worm(best)

    # --- info / title --------------------------------------------------------
    def _update_info(self):
        w = self.selected
        if w is None:
            self.info.setText("Select a worm to inspect or edit it.")
            return
        c = w.centroid(self.frame)
        lines = [f"Worm #{w.obj_id}"]
        if c is not None:
            lines.append(f"pos: ({c[0]:.0f}, {c[1]:.0f})")
            scales = self._unit_scales()
            for name in (self.area_metric, self.length_metric,
                         SPEED_METRIC, ACCEL_METRIC):
                v = w.metrics.get(name, {}).get(self.frame)
                if v is None or v != v:
                    continue
                factor, label = scales.get(name, (1.0, name))
                lines.append(f"{label}: {v * factor:.1f}")
        else:
            lines.append("(absent this frame)")
        self.info.setText("\n".join(lines))

    def _update_title(self):
        self.setWindowTitle(self._base_title + (" *" if self.masks.dirty else ""))

    # --- playback / keys -----------------------------------------------------
    def _step(self, delta):
        self._pause()
        self.set_frame(self.frame + delta)

    def _toggle_play(self):
        self._pause() if self.timer.isActive() else self._play()

    def _play(self):
        if not self.n:
            return
        if self.frame >= self.n - 1:
            self.set_frame(0)
        self.timer.start(self._interval)
        self.play_btn.setText("⏸")

    def _pause(self):
        was_active = self.timer.isActive()
        self.timer.stop()
        self.play_btn.setText("▶")
        if was_active:                       # redraw skeleton now that we stopped
            self._refresh_sel_skel()
            self._render_overlay()

    def _tick(self):
        if self.frame >= self.n - 1:
            self._pause()
            return
        self.set_frame(self.frame + 1)

    def keyPressEvent(self, e):
        ctrl = e.modifiers() & Qt.KeyboardModifier.ControlModifier
        shift = e.modifiers() & Qt.KeyboardModifier.ShiftModifier
        key = e.key()
        if ctrl and key == Qt.Key.Key_Z and not shift:
            self._undo_action()
        elif ctrl and (key == Qt.Key.Key_Y or (key == Qt.Key.Key_Z and shift)):
            self._redo_action()
        elif ctrl and key == Qt.Key.Key_S:
            self._save()
        elif key == Qt.Key.Key_P:
            self._propagate()
        elif key == Qt.Key.Key_N and self.add_btn.isEnabled():
            self._add_worm()                 # N: new worm, armed with SAM +
        elif key == Qt.Key.Key_O:
            self.onion_btn.toggle()
        elif key == Qt.Key.Key_Space:
            self._toggle_play()
        elif key == Qt.Key.Key_Left:
            # Normally handled by the window-wide shortcut (LayoutMixin.
            # _build_shortcuts) so a focused slider can't eat it; kept here for
            # key events delivered straight to the window.
            self._step(-10 if shift else -1)
        elif key == Qt.Key.Key_Right:
            self._step(10 if shift else 1)
        else:
            super().keyPressEvent(e)

    # --- window events -------------------------------------------------------
    def showEvent(self, e):
        super().showEvent(e)
        if not self._fitted and self.selected is None and self.zoomv.has_image():
            self.zoomv.frame_fit()           # sensible default framing at startup
            self._fitted = True

    def closeEvent(self, e):
        self._shutdown_sam()                 # stop the worker thread cleanly
        super().closeEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.selected is None and self.zoomv.has_image():
            self.zoomv.frame_fit()
