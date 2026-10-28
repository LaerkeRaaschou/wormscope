"""Everything the main window draws: the composited overlay, the selected
skeleton and the SAM2 point markers.

The pixel work itself lives in :mod:`wormscope.render`; this mixin only decides
*what* to composite for the current state.
"""

import numpy as np

from ..config import SKEL_RGB
from ..render import blend_all_masks, stamp_marker, thicken
from ..widgets import np_to_qimage


class RenderMixin:

    # --- compositing ---------------------------------------------------------
    def _render_overlay(self, with_extras=True):
        """Composite the original frame with every worm's mask. The Video gets
        masks only; the Zoom additionally gets the skeleton and the
        SAM2 markers. Blends only within each worm's bbox."""
        if not self.n:                       # no video loaded: nothing to draw
            return
        sel_id = self.selected.obj_id if self.selected else None
        raw = self.frames.get(self.frame)
        base = raw.copy()                                # uint8, don't touch cache
        blend_all_masks(base, self.worms, self.masks, self.frame, sel_id,
                        self.outline_only)
        self._overlay = base
        self.video.set_frame(np_to_qimage(base))         # video: raw + masks only

        zoom_arr, owned = base, False    # the zoom shares the video's pixels

        extras = with_extras and self.selected is not None and not self.timer.isActive()
        if extras and self._sel_skel is not None and self._sel_skel.any():
            if not owned:
                zoom_arr, owned = base.copy(), True
            zoom_arr[thicken(self._sel_skel)] = np.array(SKEL_RGB, np.uint8)
        if extras:                                   # SAM2 point markers
            pts = self._sam_points.get((self.selected.obj_id, self.frame))
            if pts:
                if not owned:
                    zoom_arr, owned = base.copy(), True
                for px, py, lab in pts:
                    col = (40, 200, 40) if lab == 1 else (225, 40, 40)
                    stamp_marker(zoom_arr, px, py, col)
        h, w = base.shape[:2]
        self.zoomv.set_image(np_to_qimage(zoom_arr), w, h)

    def _refresh_sel_skel(self):
        """Selected worm's skeleton for the current frame (from skel/ or a live
        recompute), skipped during playback to keep stepping snappy."""
        self._sel_skel = None
        if self.selected is None or self.timer.isActive():
            return
        self._sel_skel = self.masks.skel(self.selected.obj_id, self.frame)

    # --- overlay toggles / framing -------------------------------------------
    def _toggle_outline(self, on):
        self.outline_only = bool(on)
        self._render_overlay()

    def _fit_view(self):
        bbox = self.selected.bbox(self.frame) if self.selected else None
        if bbox is not None:
            self.zoomv.frame_bbox(bbox)
        else:
            self.zoomv.frame_fit()

