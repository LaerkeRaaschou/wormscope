"""Managing the worm list: add, delete and reassign IDs (all undoable)."""

from pathlib import Path

from PyQt6.QtWidgets import QInputDialog, QMessageBox

from .. import config
from ..core.models import Worm
from ..widgets import ZoomView


class WormListMixin:

    # --- list bookkeeping ----------------------------------------------------
    def _insert_worm(self, w, row=None):
        if row is None:
            row = len(self.worms)
        self.worms.insert(row, w)
        self.by_id[w.obj_id] = w
        self.list.blockSignals(True)
        self.list.insertItem(row, f"#{w.obj_id}")
        self.list.blockSignals(False)

    def _remove_worm_obj(self, w):
        if w not in self.worms:
            return
        row = self.worms.index(w)
        self.worms.pop(row)
        self.by_id.pop(w.obj_id, None)
        self.list.blockSignals(True)
        self.list.takeItem(row)
        self.list.blockSignals(False)
        if self.selected is w:
            self.selected = None
            self.list.setCurrentRow(-1)
            self.plot.set_worm(None)
            self._sel_skel = None
            self._render_overlay()
            self._update_info()

    # --- actions -------------------------------------------------------------
    def _add_worm(self):
        new_id = (max(self.by_id) + 1) if self.by_id else 0
        w = Worm(obj_id=new_id)
        self._insert_worm(w)
        self.select_worm(w)
        self._arm_new_worm_tool(new_id)
        self._push_action(lambda: self._remove_worm_obj(w),
                          lambda: (self._insert_worm(w), self.select_worm(w)))

    def _arm_new_worm_tool(self, new_id):
        """Pick the tool a brand-new worm starts with.

        SAM2 by default (NEW_WORM_TOOL in config.py): a click on the animal
        usually beats drawing its outline by hand. It needs the frames on disk,
        though, so an in-memory dataset (the demo, and anything opened without a
        frames/ folder) falls back to the brush rather than arming a tool whose
        first click can only raise a dialog."""
        use_sam = config.NEW_WORM_TOOL == "sam" and self._frames_dir is not None
        if use_sam:
            self.btn_sam_pos.setChecked(True)
            self._set_mode(ZoomView.MODE_SAM_POS)
            self.info.setText(f"Worm #{new_id}\nClick on the worm to segment "
                              f"it.\nShift+click marks what is NOT the worm.")
        else:
            self.btn_draw.setChecked(True)
            self._set_mode(ZoomView.MODE_DRAW)
            self.info.setText(f"Worm #{new_id}\nDraw its mask on this frame.")

    def _rename_sam_state(self, old, new):
        """Move a worm's corrected / prompted frames to its new ID.

        SAM2 keeps its prompts per ID, and the old ID's session can't be
        renamed, so it is dropped: every frame that was prompted or painted
        becomes a hand edit of the new ID. The next run re-sends those masks
        as prompts, and “lock edits” keeps protecting them."""
        moved = {(new, i) for (o, i) in self._hand_edited | self._anchored
                 if o == old}
        self._hand_edited = {k for k in self._hand_edited if k[0] != old} | moved
        self._anchored = {k for k in self._anchored if k[0] != old}
        self._sam_points = {k: v for k, v in self._sam_points.items()
                            if k[0] != old}
        if old in self._prompted_worms:
            self._prompted_worms.discard(old)
            self._prompted_worms.add(new)
        if self.sam is not None:
            self.sam.forget(old)

    def _delete_worm(self):
        w = self.selected
        if w is None:
            return
        if QMessageBox.question(self, "Delete worm",
                                f"Delete worm #{w.obj_id} from all frames?") \
                != QMessageBox.StandardButton.Yes:
            return
        masks, paths, files = self.masks.pop_worm(w.obj_id)
        self._pending_delete.update(files)
        if self.sam is not None:             # its SAM2 session goes with it
            self.sam.forget(w.obj_id)
        self._hand_edited = {k for k in self._hand_edited if k[0] != w.obj_id}
        self._anchored = {k for k in self._anchored if k[0] != w.obj_id}
        self._remove_worm_obj(w)
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()

        def undo():
            self.masks.restore_worm(masks, paths)
            self._pending_delete.difference_update(files)
            self._insert_worm(w)
            self.select_worm(w)
            self._update_title()

        def redo():
            self.masks.pop_worm(w.obj_id)
            self._pending_delete.update(files)
            self._remove_worm_obj(w)
            self._update_title()
        self._push_action(undo, redo)

    def _reassign_id(self):
        w = self.selected
        if w is None:
            return
        val, ok = QInputDialog.getInt(self, "Reassign worm ID",
                                      "New ID:", w.obj_id, 0, 9999)
        if not ok or val == w.obj_id:
            return
        if val in self.by_id:
            QMessageBox.warning(self, "ID in use",
                                f"Worm #{val} already exists. Merging isn't "
                                f"supported — pick a free ID.")
            return
        old = w.obj_id
        self._do_reassign(w, old, val)
        self._push_action(lambda: self._do_reassign(w, val, old),
                          lambda: self._do_reassign(w, old, val))

    def _do_reassign(self, w, old, new):
        old_files = self.masks.rename_worm(old, new)
        self._rename_sam_state(old, new)
        # old-named files on disk are now garbage; but we're about to (re)write
        # files under `new`, so cancel any pending deletion of that name.
        self._pending_delete = {p for p in self._pending_delete
                                if f"worm{new:02d}_" not in Path(p).name}
        self._pending_delete.update(old_files)
        self.by_id.pop(old, None)
        w.obj_id = new
        self.by_id[new] = w
        row = self.worms.index(w)
        self.list.item(row).setText(f"#{new}")
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()
        self.select_worm(w)
