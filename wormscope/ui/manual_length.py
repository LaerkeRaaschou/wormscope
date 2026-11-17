"""Manual length: place points along a worm to measure its length by hand.

The "Measure" tool (L) works on the selected worm in the Zoom view::

    click                add a point at the end of the line
    click on the line    insert a point into that segment
    drag a point         move it
    Shift+click a point  remove it
    Backspace / Delete   remove the last point
    Clear                remove this worm's line on this frame

One line per worm per frame, stored on the worm (``Worm.manual``). It needs no
mask, so a video can be measured by hand first and segmented with SAM2 later:
the worm IDs in the list are what the two are matched on. Every change is
undoable, and Save writes manual_lengths.csv (see io/manual_lengths.py). The
manual lengths also appear as orange diamonds on the length plot.
"""

import math

from ..config import ANNOTATION_MANUAL
from ..core.manual import polyline_length
from ..widgets import ZoomView


class ManualLengthMixin:

    # --- state ---------------------------------------------------------------
    def _manual_points(self, worm=None, idx=None):
        worm = self.selected if worm is None else worm
        idx = self.frame if idx is None else idx
        if worm is None:
            return []
        return worm.manual.get(idx, [])

    def _set_manual(self, worm, idx, pts):
        """Set (or, with no points, remove) a worm's line on one frame."""
        if pts:
            worm.manual[idx] = [tuple(p) for p in pts]
        else:
            worm.manual.pop(idx, None)
        self._mark_manual_dirty()

    def _mark_manual_dirty(self):
        self._manual_dirty = True
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()

    def _change_manual(self, worm, idx, after):
        """Replace the line with `after` as one undoable step."""
        before = list(worm.manual.get(idx, []))
        after = list(after)
        if before == after:
            return
        self._set_manual(worm, idx, after)
        self._after_manual_edit()
        self._push_action(lambda: self._restore_manual(worm, idx, before),
                          lambda: self._restore_manual(worm, idx, after))

    def _restore_manual(self, worm, idx, pts):
        """Undo / redo: put a line back and show it (switching to Manual mode
        if needed — an undo you can't see is an undo you can't check)."""
        if not self.manual_mode:
            self._set_annotation_mode(ANNOTATION_MANUAL)
        self._set_manual(worm, idx, pts)
        if worm is not self.selected and worm in self.worms:
            self.select_worm(worm)
        self.set_frame(idx)
        self._after_manual_edit()

    def _after_manual_edit(self):
        self._refresh_measure_view()
        self.plot.set_worm(self.selected)
        self.plot.set_frame(self.frame)
        self._update_info()

    # --- the tool ------------------------------------------------------------
    def _arm_measure_tool(self):
        if not self.manual_mode:
            self._set_annotation_mode(ANNOTATION_MANUAL)
            return                           # switching arms the tool
        self.btn_measure.setChecked(True)
        self._set_mode(ZoomView.MODE_MEASURE)

    def _measure_ready(self):
        """True when there is a worm to measure; otherwise say what to do."""
        if self.selected is None:
            self.info.setText("Select a worm (or Add one) first, then click "
                              "points along it with Measure.")
            return False
        return True

    def _on_measure_added(self, x, y, insert_at):
        if not self._measure_ready():
            return
        pts = list(self._manual_points())
        if insert_at < 0:
            pts.append((x, y))
        else:
            pts.insert(insert_at, (x, y))
        self._change_manual(self.selected, self.frame, pts)

    def _on_measure_removed(self, i):
        pts = list(self._manual_points())
        if self.selected is None or not 0 <= i < len(pts):
            return
        del pts[i]
        self._change_manual(self.selected, self.frame, pts)

    def _on_measure_moved(self, i, x, y):
        """A point is being dragged: move it live; one undo step on release."""
        if self.selected is None:
            return
        pts = self.selected.manual.get(self.frame)
        if not pts or not 0 <= i < len(pts):
            return
        if self._measure_drag is None:
            self._measure_drag = (self.selected, self.frame, list(pts))
        pts[i] = (x, y)
        self._refresh_measure_view()
        self._update_info()

    def _on_measure_move_finished(self):
        if self._measure_drag is None:
            return
        worm, idx, before = self._measure_drag
        self._measure_drag = None
        after = list(worm.manual.get(idx, []))
        if after == before:
            return
        self._mark_manual_dirty()
        self._after_manual_edit()
        self._push_action(lambda: self._restore_manual(worm, idx, before),
                          lambda: self._restore_manual(worm, idx, after))

    def _measure_remove_last(self):
        pts = list(self._manual_points())
        if self.selected is None or not pts:
            return
        self._change_manual(self.selected, self.frame, pts[:-1])

    def _measure_clear(self):
        if self.selected is None:
            return
        self._change_manual(self.selected, self.frame, [])

    # --- display -------------------------------------------------------------
    def _manual_text(self, pts):
        """'412.7 px' or '165.1 µm' (with the point count when asked)."""
        length = polyline_length(pts)
        if self.px_to_um:
            return f"{length * self.px_to_um:.1f} µm"
        return f"{length:.1f} px"

    def _refresh_measure_view(self):
        """Draw the manual lines of this frame on both canvases."""
        if not self.n:
            return
        if not self.manual_mode:             # Automatic mode: lines hidden
            self.zoomv.set_measure([], [], "")
            self.video.set_measure_lines([])
            return
        sel = self.selected
        active = self._manual_points() if sel is not None else []
        others = [w.manual[self.frame] for w in self.worms
                  if w is not sel and self.frame in w.manual]
        label = self._manual_text(active) if len(active) >= 2 else ""
        self.zoomv.set_measure(active, others, label)
        self.video.set_measure_lines([(p, False) for p in others]
                                     + ([(active, True)] if active else []))
        n = len(active)
        self.measure_label.setText(
            (self._manual_text(active) if n >= 2 else "—")
            + (f"  ({n} pt{'s' if n != 1 else ''})" if n else ""))

    def _manual_info_line(self, worm):
        pts = worm.manual.get(self.frame, [])
        if len(pts) >= 2:
            return f"manual length: {self._manual_text(pts)} ({len(pts)} pts)"
        if pts:
            return "manual length: 1 point — add more along the worm"
        return None

    def _worm_at_manual_line(self, x, y, radius):
        """The worm whose manual line on this frame passes nearest (x, y),
        within `radius` image pixels — so a worm measured only by hand can be
        picked on the Video too."""
        best, best_d = None, radius
        for w in self.worms:
            pts = w.manual.get(self.frame, [])
            for (x0, y0), (x1, y1) in zip(pts, pts[1:] or pts):
                dx, dy = x1 - x0, y1 - y0
                L2 = dx * dx + dy * dy
                t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / L2))
                d = math.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
                if d < best_d:
                    best, best_d = w, d
        return best
