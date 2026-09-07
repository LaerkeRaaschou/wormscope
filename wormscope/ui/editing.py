"""Mask editing in the Zoom canvas: tools, brush strokes and metric refresh."""

from ..core.geometry import stamp_disk, stamp_stroke
from ..core.kinematics import refresh_kinematics
from ..core.metrics import apply_mask_stats
from ..widgets import ZoomView


class EditingMixin:

    # --- tool state ----------------------------------------------------------
    def _set_mode(self, mode):
        self.zoomv.mode = mode
        if mode != ZoomView.MODE_PAN and self.selected is None:
            self.info.setText("Pick a worm first, then Draw / Erase in the Zoom view.")

    def _set_brush(self, v):
        self.zoomv.brush = v
        self.brush_val.setText(str(v))

    def _set_zoom(self, v):
        self.zoomv.set_zoom(v / 10.0)

    def _on_viewport_changed(self):
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(round(self.zoomv.zoom * 10)))
        self.zoom_slider.blockSignals(False)
        self.video.set_viewport(self.zoomv.viewport_rect())

    # --- painting ------------------------------------------------------------
    def _on_paint(self, x, y):
        """One mouse position inside a stroke: stamp the brush and redraw."""
        if self.selected is None:
            return
        # paint_mode(), not mode: a Shift-stroke borrows the opposite tool.
        value = (self.zoomv.paint_mode() == ZoomView.MODE_DRAW)
        mask = self.masks.get(self.selected.obj_id, self.frame)
        r = self.zoomv.brush
        if self._last_paint is None:
            self._stroke_before = mask.copy()             # snapshot for undo
            self._stroke_key = (self.selected.obj_id, self.frame)
            stamp_disk(mask, x, y, r, value)
        else:
            x0, y0 = self._last_paint
            stamp_stroke(mask, x0, y0, x, y, r, value)    # no gaps when dragging
        self._last_paint = (x, y)
        self._render_overlay(with_extras=False)       # fast live feedback

    def _on_stroke_finished(self):
        """Mouse released: recompute metrics once and push a single undo step."""
        self._last_paint = None
        if self.selected is None or self._stroke_before is None:
            return
        wid, idx = self._stroke_key
        before, after = self._stroke_before, self.masks.get(wid, idx).copy()
        self._stroke_before = None
        self.masks.set_dirty(wid, idx)
        # Remember that a human painted this frame: before any re-track, it is
        # handed to SAM2 as a mask prompt so the correction survives and the
        # tracker works *from* it (see SamMixin._anchor_edits).
        self._hand_edited.add((wid, idx))
        if self.sam is not None and self.sam.busy:
            # SAM2 is tracking right now: this frame is yours, and the results
            # arriving for it will be dropped rather than overwrite your work.
            self._edited_during_run.add((wid, idx))
        self._recompute(self.selected, self.frame)
        self._after_edit()
        self._push_action(lambda: self._restore_mask(wid, idx, before),
                          lambda: self._restore_mask(wid, idx, after))

    def _after_edit(self):
        """Refresh everything after the selected worm's current mask changed."""
        self._refresh_sel_skel()
        self._render_overlay()
        self.plot.set_worm(self.selected)
        self.plot.set_frame(self.frame)
        self._update_info()
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()

    # --- metrics -------------------------------------------------------------
    def _apply_mask_stats(self, worm, idx):
        """Recompute area / length / centroid / bbox for one worm-frame from its
        mask (no kinematics — callers do that once)."""
        apply_mask_stats(worm, self.masks, idx, self.area_metric,
                         self.length_metric)

    def _recompute(self, worm, idx):
        self._apply_mask_stats(worm, idx)
        self._recompute_kinematics(worm)     # centroid moved -> speed/accel change

    def _recompute_kinematics(self, worm):
        """(Re)fill the worm's speed + acceleration series from its centroids."""
        refresh_kinematics(worm)

    def _refresh_plots(self):
        """Recompute every metric (area, length, speed, acceleration) for all
        worms from the current masks, then redraw the plots."""
        for w in self.worms:
            for idx in self.masks.frames_of(w.obj_id):
                self._apply_mask_stats(w, idx)
            self._recompute_kinematics(w)
        self.plot.set_worm(self.selected)
        self.plot.set_frame(self.frame)
        self._render_overlay()
        self._update_info()
        print(f"[refresh] recomputed metrics for {len(self.worms)} worms")

    def _toggle_smooth(self, on):
        self.plot.set_smooth(on)
        self.plot.set_frame(self.frame)

    # --- copy a mask forward -------------------------------------------------
    def _propagate(self):
        """Copy the selected worm's current mask into the next frame."""
        w = self.selected
        if w is None or self.frame >= self.n - 1:
            return
        nxt = self.frame + 1
        before = self.masks.get(w.obj_id, nxt).copy()
        after = self.masks.get(w.obj_id, self.frame).copy()
        self.masks.set(w.obj_id, nxt, after.copy())
        self._hand_edited.add((w.obj_id, nxt))
        self._recompute(w, nxt)
        self.set_frame(nxt)                              # step onto the new frame
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()
        self._push_action(lambda: self._restore_mask(w.obj_id, nxt, before),
                          lambda: self._restore_mask(w.obj_id, nxt, after))
