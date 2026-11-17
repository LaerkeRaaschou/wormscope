"""Automatic / Manual: which kind of annotation the window works with.

One video folder holds both kinds of data side by side — the SAM2 / brush
masks with their metrics (binary/, skel/, measurements.csv) and the manual
length lines (manual_lengths.csv) — and Save always writes both. The mode
only decides what you *see and edit*:

    Automatic   masks, skeleton, SAM2 points, area/length/speed/acceleration;
                the AI-edit and brush tools. Manual lines are hidden.
    Manual      the raw frames with the manual length lines; the Measure tool.
                Masks, skeletons and their metrics are hidden.

The worm list always shows every worm, whichever data it has: selecting one
moves the Zoom view to where it is (its mask or its manual line, on this frame
or the nearest frame that has either) without showing the hidden data.

The mode is remembered per video in wormscope.json, and the last one used is
the starting mode for a video opened for the first time.
"""

from .. import settings
from ..config import ANNOTATION_AUTOMATIC, ANNOTATION_MANUAL, MANUAL_BBOX_PAD
from ..widgets import ZoomView


class AnnotationModeMixin:

    # --- queries -------------------------------------------------------------
    @property
    def manual_mode(self) -> bool:
        return self.annotation_mode == ANNOTATION_MANUAL

    def _initial_annotation_mode(self, session):
        mode = (session or {}).get("annotation_mode")
        if mode in (ANNOTATION_AUTOMATIC, ANNOTATION_MANUAL):
            return mode
        has_masks = any(w.bboxes for w in self.worms)
        has_manual = any(w.manual for w in self.worms)
        if has_manual and not has_masks:
            return ANNOTATION_MANUAL
        if has_masks:
            return ANNOTATION_AUTOMATIC
        last = settings.get("annotation_mode")
        return last if last in (ANNOTATION_AUTOMATIC, ANNOTATION_MANUAL) \
            else ANNOTATION_AUTOMATIC

    # --- switching -----------------------------------------------------------
    def _set_annotation_mode(self, mode, remember=True):
        """Show / hide the tools and data of the two kinds of annotation."""
        if mode not in (ANNOTATION_AUTOMATIC, ANNOTATION_MANUAL):
            return
        changed = mode != getattr(self, "annotation_mode", None)
        self.annotation_mode = mode
        manual = mode == ANNOTATION_MANUAL
        for b, m in ((self.mode_auto_btn, ANNOTATION_AUTOMATIC),
                     (self.mode_manual_btn, ANNOTATION_MANUAL)):
            b.blockSignals(True)
            b.setChecked(m == mode)
            b.blockSignals(False)
        # the tools of the other mode disappear rather than grey out
        self.ai_box.setVisible(not manual)
        self.brush_box.setVisible(not manual)
        self.outline_btn.setVisible(not manual)
        self.measure_box.setVisible(manual)
        if self.n:
            self.export_btn.setEnabled(not manual)   # it renders the masks
            self.refresh_btn.setEnabled(not manual)  # recomputes mask metrics
        self.plot.set_view(show_auto=not manual, show_manual=manual)
        # the active tool must belong to this mode
        if manual:
            if self.n and self.selected is not None:
                self._arm_measure_tool()
            else:
                self._pan_tool()
        elif self.zoomv.mode == ZoomView.MODE_MEASURE:
            self._pan_tool()
        if remember and changed:
            settings.save({"annotation_mode": mode})
        if not manual and self.n and changed:
            self._start_sam(warm=True)       # SAM2 is only needed here
        if self.n:
            self._refresh_sel_skel()
            self._render_overlay()
            self.plot.set_worm(self.selected)
            self.plot.set_frame(self.frame)
            self._update_info()

    def _pan_tool(self):
        self.btn_pan.setChecked(True)
        self.zoomv.mode = ZoomView.MODE_PAN

    # --- where is a worm? ----------------------------------------------------
    def _manual_bbox(self, worm, frame):
        pts = worm.manual.get(frame)
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = MANUAL_BBOX_PAD
        return (int(min(xs)) - pad, int(min(ys)) - pad,
                int(max(xs)) + 1 + pad, int(max(ys)) + 1 + pad)

    def _locate_worm(self, worm, frame=None):
        """A bbox to frame the worm in the Zoom view, from whatever data it
        has — visible in this mode or not: its mask or manual line on
        `frame`, else on the nearest frame that has either. None when the
        worm has no data at all (a brand-new worm)."""
        if worm is None:
            return None
        frame = self.frame if frame is None else frame

        def at(f):
            first, second = ((self._manual_bbox(worm, f), worm.bbox(f))
                             if self.manual_mode else
                             (worm.bbox(f), self._manual_bbox(worm, f)))
            return first if first is not None else second

        box = at(frame)
        if box is not None:
            return box
        frames = set(worm.bboxes) | {f for f, p in worm.manual.items() if p}
        if not frames:
            return None
        return at(min(frames, key=lambda f: (abs(f - frame), f)))
