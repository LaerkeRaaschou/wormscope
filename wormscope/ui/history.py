"""Undo / redo.

Every reversible action pushes a ``(undo_fn, redo_fn)`` pair; both closures must
leave the GUI in a consistent, visible state (the mask helpers below do that for
you). Keep new actions symmetric — whatever redo does, undo must exactly undo.
"""

from ..config import ANNOTATION_AUTOMATIC, UNDO_LIMIT


class HistoryMixin:

    def _push_action(self, undo_fn, redo_fn):
        self._undo.append((undo_fn, redo_fn))
        if len(self._undo) > UNDO_LIMIT:
            self._undo.pop(0)
        self._redo.clear()
        self._sync_undo_buttons()

    def _undo_action(self):
        if not self._undo:
            return
        u, r = self._undo.pop()
        u()
        self._redo.append((u, r))
        self._sync_undo_buttons()

    def _redo_action(self):
        if not self._redo:
            return
        u, r = self._redo.pop()
        r()
        self._undo.append((u, r))
        self._sync_undo_buttons()

    def _sync_undo_buttons(self):
        self.undo_btn.setEnabled(bool(self._undo))
        self.redo_btn.setEnabled(bool(self._redo))

    def _restore_mask(self, wid, idx, arr):
        """Set a worm's mask for one frame and show the result (for undo/redo)."""
        if self.manual_mode:                 # masks are only shown in Automatic
            self._set_annotation_mode(ANNOTATION_AUTOMATIC)
        self.masks.set(wid, idx, arr.copy())
        w = self.by_id.get(wid)
        if w is not None:
            self._recompute(w, idx)
            if w is not self.selected:
                self.select_worm(w)
        self.set_frame(idx)
        self.plot.set_worm(self.selected)
        self.plot.set_frame(self.frame)
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()
