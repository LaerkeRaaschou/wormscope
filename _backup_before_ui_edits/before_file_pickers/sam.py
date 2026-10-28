"""SAM2 point-prompt segmentation, wired to the SAM +/− tools.

Everything here is asynchronous: prompts and tracking are queued on the worker
thread in :mod:`wormscope.ui.sam_worker`, and results arrive as Qt signals. The
brush, undo, frame stepping and the plots all keep working while SAM2 thinks —
the only things that wait are further SAM2 requests, which queue behind the
running one.

The config + checkpoint are asked for once, the first time SAM2 is needed, and
remembered in ~/.wormscope/settings.json — see :mod:`wormscope.settings` for how
to change them later without the GUI. The backend itself is swappable — see
:mod:`wormscope.segmentation`.
"""

from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from .. import segmentation, settings
from ..core.models import Worm
from ..segmentation import hydra_config_name, pair_warning, sam2_config_dir
from .sam_worker import SamWorker


class SamMixin:

    # --- where SAM2's files live --------------------------------------------
    def _choose_sam2_files(self, explain=True):
        """Ask for the SAM2 config + checkpoint once and remember them.

        Returns True if both were picked and saved."""
        cfg, ckpt, _ = settings.sam2_paths()
        if explain and not (cfg and ckpt):
            if QMessageBox.question(
                    self, "SAM2 files",
                    "SAM2 needs its config (.yaml) and checkpoint (.pt).\n\n"
                    "Pick them once and wormscope will remember them for every "
                    "future launch.\n\nLocate them now?") \
                    != QMessageBox.StandardButton.Yes:
                return False
        # Start where the configs actually live: next to the current choice, or
        # in the installed sam2 package's own configs/ folder.
        pkg_cfg = sam2_config_dir()
        start_cfg = str(Path(cfg).parent) if cfg else \
            (str(pkg_cfg) if pkg_cfg else str(Path.home()))
        new_cfg, _f = QFileDialog.getOpenFileName(
            self, "Select the SAM2 config (.yaml, usually inside the sam2 "
                  "package's configs/ folder)", start_cfg,
            "SAM2 config (*.yaml *.yml);;All files (*)")
        if not new_cfg:
            return False
        start_ckpt = str(Path(ckpt).parent) if ckpt else str(Path(new_cfg).parent)
        new_ckpt, _f = QFileDialog.getOpenFileName(
            self, "Select the SAM2 checkpoint", start_ckpt,
            "SAM2 checkpoint (*.pt *.pth);;All files (*)")
        if not new_ckpt:
            return False
        path = settings.save_sam2_paths(new_cfg, new_ckpt)
        self._shutdown_sam()                 # rebuild with the new paths
        name, overrides = hydra_config_name(new_cfg)
        notes = ""
        if overrides:
            notes += ("\n\nNote: this config lives outside the sam2 package, so "
                      "its folder is added to Hydra's search path at load time.")
        mismatch = pair_warning(name, new_ckpt)
        if mismatch:
            notes += f"\n\n⚠ {mismatch}"
        QMessageBox.information(
            self, "SAM2 files",
            f"Saved to:\n{path}\n\nconfig:\n{new_cfg}\n(loaded by Hydra as "
            f"'{name}')\n\ncheckpoint:\n{new_ckpt}{notes}"
            f"\n\nYou won't be asked again. To change them later:\n"
            f"  python -m wormscope.settings --clear-sam2\n"
            f"(or --set-sam2 <config> <checkpoint>)")
        self._start_sam(warm=True)
        return True

    # --- the worker ----------------------------------------------------------
    def _start_sam(self, warm=False):
        """Spin up the SAM2 worker thread if it isn't running.

        With `warm`, the model is loaded and the first frame encoded straight
        away in the background: that is most of the cost of the first click, so
        doing it while you are still looking at the video makes that click feel
        immediate. Returns False if SAM2 cannot run here at all."""
        if self.sam is not None:
            return True
        if self._frames_dir is None or not self.n:
            return False
        self.sam = SamWorker(segmentation.make_segmenter, self.frames,
                             self.frames.get(0).shape[:2], parent=self)
        self.sam.built.connect(self._on_sam_built)
        self.sam.needsFiles.connect(self._on_sam_needs_files)
        self.sam.promptDone.connect(self._on_prompt_done)
        self.sam.frameTracked.connect(self._on_frame_tracked)
        self.sam.jobProgress.connect(self._on_sam_progress)
        self.sam.jobDone.connect(self._on_sam_job_done)
        self.sam.jobFailed.connect(self._on_sam_job_failed)
        self.sam.start()
        if warm:
            cfg, ckpt, _src = settings.sam2_paths()
            if cfg and ckpt:                 # silent: nothing to ask the user
                self.progress_panel.start("Preparing SAM2", 0,
                                          "loading the model…")
                self.sam.build()
                self.sam.prewarm([0])
        return True

    def _shutdown_sam(self):
        if self.sam is not None:
            self.sam.shutdown()
            self.sam = None
            self._sam_ready = False
        self._point_jobs.clear()             # their jobs died with the worker

    def _ensure_segmenter(self, explain=True):
        """Make sure the worker exists and SAM2 has files to load.

        Returns False (after explaining) when SAM2 cannot run at all. True means
        "queued or ready" — the caller's request runs when the worker reaches
        it, not necessarily now."""
        if self._frames_dir is None:
            if explain:
                QMessageBox.warning(
                    self, "SAM2",
                    "SAM2 needs the frames on disk. Use “Upload new video frames…” to "
                    "start from a folder of images.")
            return False
        # No path check here on purpose: another backend may need no files at
        # all. If SAM2 does and they are missing, the worker says so and we
        # answer with the picker (_on_sam_needs_files).
        return self._start_sam()

    # --- results coming back from the worker ---------------------------------
    def _on_sam_built(self, ok, message):
        self._sam_ready = bool(ok)
        if ok:
            print("[sam2] worker ready")
            return
        self.progress_panel.finish()
        cfg, ckpt, source = settings.sam2_paths()
        name, _ov = hydra_config_name(cfg) if cfg else ("?", [])
        mismatch = pair_warning(name, ckpt) if cfg and ckpt else ""
        # Offer the picker right here: with the toolbar button gone, this
        # dialog is where a wrong or moved pair gets corrected.
        answer = QMessageBox.question(
            self, "SAM2 unavailable",
            f"Could not start SAM2:\n\n{message}\n\n"
            + (f"⚠ {mismatch}\n\n" if mismatch else "")
            + f"In use ({source}):\nconfig: {name}\ncheckpoint: {ckpt}"
            f"\n\nPick a different config + checkpoint now?\n"
            f"(If SAM2 itself is missing: pip install sam2)")
        self._shutdown_sam()
        if answer == QMessageBox.StandardButton.Yes:
            self._choose_sam2_files(explain=False)

    def _on_sam_needs_files(self, _message):
        """SAM2 has no config/checkpoint yet: ask once, then it is remembered."""
        self.progress_panel.finish()
        self._shutdown_sam()
        if self._choose_sam2_files():
            self.info.setText("SAM2 is loading — click again in a moment.")
        else:
            self.info.setText("SAM2 needs a config and a checkpoint.\n"
                              "You will be asked again next time you click a "
                              "SAM point.")

    def _on_prompt_done(self, oid, frame, mask, kind):
        """A point or mask prompt came back with SAM2's version of the mask.

        For a point click this is one undo step covering both the point and the
        mask it produced: undo removes the marker, takes the point out of
        SAM2's session and puts the old mask back."""
        job = None
        if kind == "points" and self._point_jobs:
            job = self._point_jobs.popleft()  # jobs run in order: FIFO matches
            if job[:2] != (oid, frame):
                job = None
        if mask is None:
            return
        before = self.masks.get(oid, frame).copy()
        self.masks.set(oid, frame, np.asarray(mask, bool))
        self._anchored.add((oid, frame))
        self._hand_edited.discard((oid, frame))
        w = self.by_id.get(oid)
        if w is not None:
            self._recompute(w, frame)
        after = self.masks.get(oid, frame).copy()
        if job is None:                      # a mask prompt
            self._push_action(lambda: self._restore_mask(oid, frame, before),
                              lambda: self._restore_mask(oid, frame, after))
        else:
            _o, _f, pts_before, pts_after, was_prompted, was_anchored, was_hand = job
            self._push_action(
                lambda: self._restore_point_step(oid, frame, before, pts_before,
                                                 was_prompted, was_anchored,
                                                 was_hand),
                lambda: self._restore_point_step(oid, frame, after, pts_after,
                                                 True, True, False))
        self._after_edit()

    def _restore_point_step(self, oid, frame, mask, points, prompted, anchored,
                            hand_edited):
        """Undo/redo of one SAM click: the frame's points, SAM2's copy of
        them, the bookkeeping sets and the mask all go back together."""
        key = (oid, frame)
        if points:
            self._sam_points[key] = list(points)
        else:
            self._sam_points.pop(key, None)
        (self._anchored.add if anchored else self._anchored.discard)(key)
        (self._hand_edited.add if hand_edited else self._hand_edited.discard)(key)
        (self._prompted_worms.add if prompted else self._prompted_worms.discard)(oid)
        if self.sam is not None:
            self.sam.set_points(frame, oid, [(p[0], p[1]) for p in points],
                                [p[2] for p in points])
        self._restore_mask(oid, frame, mask)  # also redraws the markers

    def _protected(self, oid, frame):
        """True when a tracking result for this frame must be discarded.

        Always: frames you edited *while the run was going* — your hand beats
        the tracker's guess. With “lock edits” ticked: every frame you have
        corrected by hand, ever. Those are still sent to SAM2 as prompts, so
        they keep steering the track; they just don't get written over."""
        if (oid, frame) in self._edited_during_run:
            return True
        return self.lock_box.isChecked() and (
            (oid, frame) in self._anchored or (oid, frame) in self._hand_edited)

    def _on_frame_tracked(self, oid, frame, mask):
        """One frame of a tracking run."""
        if self._protected(oid, frame):
            self._kept_locked.add((oid, frame))
            return
        self.masks.set(oid, frame, np.asarray(mask, bool))
        self._tracked.add(oid)

    def _on_sam_progress(self, done, total, detail):
        self.progress_panel.advance(done, detail, total=total)
        if self.progress_panel.cancelled and self.sam is not None:
            self.sam.stop_current()

    def _on_sam_job_done(self, kind, payload):
        if kind == "track":
            self._finish_tracking(payload)
        elif self.sam is None or not self.sam.busy:
            self.progress_panel.finish()

    def _on_sam_job_failed(self, kind, message):
        self.progress_panel.finish()
        if kind == "points" and self._point_jobs:
            self._point_jobs.popleft()       # that click produced no mask
        if kind in ("points", "mask", "track", "prewarm"):
            QMessageBox.warning(self, "SAM2", f"{kind} failed:\n\n{message}")

    # --- placing prompts -----------------------------------------------------
    def _on_sam_point(self, x, y, label):
        """A positive/negative click for the selected worm on this frame."""
        if self.selected is None:
            self.info.setText("Select or Add a worm first, then click SAM points.")
            return
        if not self._ensure_segmenter():
            return
        oid = self.selected.obj_id
        key = (oid, self.frame)
        before = list(self._sam_points.get(key, []))
        pts = self._sam_points.setdefault(key, [])
        pts.append((float(x), float(y), int(label)))
        # Remembered until SAM2 answers, so the point and its mask become ONE
        # undo step (see _on_prompt_done).
        self._point_jobs.append((oid, self.frame, before, list(pts),
                                 oid in self._prompted_worms,
                                 key in self._anchored,
                                 key in self._hand_edited))
        self._render_overlay()               # the marker appears immediately
        self._prompted_worms.add(oid)
        self.progress_panel.start("SAM2", 0, f"worm {oid} — frame {self.frame}")
        self.sam.add_points(self.frame, oid, [(p[0], p[1]) for p in pts],
                            [p[2] for p in pts])

    def _sam_mask_prompt(self):
        """Hand the selected worm's current (brush-corrected) mask to SAM2 as
        the prompt for this frame."""
        w = self.selected
        if w is None:
            self.info.setText("Select a worm first, then correct its mask.")
            return
        mask = self.masks.get(w.obj_id, self.frame)
        if not mask.any():
            QMessageBox.information(
                self, "Mask → prompt",
                "This worm has no mask on this frame yet. Draw one with the "
                "brush (or click SAM points) first.")
            return
        if not self._ensure_segmenter():
            return
        self._prompted_worms.add(w.obj_id)
        self.progress_panel.start("SAM2", 0,
                                  f"worm {w.obj_id} — mask on frame {self.frame}")
        self.sam.add_mask(self.frame, w.obj_id, mask.copy())
        self.info.setText(f"Worm #{w.obj_id}\nMask sent to SAM2 as the prompt "
                          f"for frame {self.frame}.\nNow “Re-segment ▶ here”.")

    # --- tracking ------------------------------------------------------------
    def _sam_resegment(self):
        """Re-track the selected worm from the current frame."""
        if self.selected is None:
            self.info.setText("Select the worm you corrected first.")
            return
        self._sam_propagate(start=self.frame,
                            also_backwards=self.back_box.isChecked(),
                            worms=[self.selected.obj_id])

    def _pending_anchors(self, oid):
        """(frame, mask) for every frame of this worm painted by hand.

        A painted frame is invisible to the tracker until it is a prompt, so a
        re-track would simply overwrite it. Sent as mask prompts first, these
        become anchors the tracker works outwards from — which is what makes
        corrections on several frames reinforce each other."""
        out = []
        for idx in sorted(idx for (w, idx) in self._hand_edited if w == oid):
            mask = self.masks.get(oid, idx)
            if mask.any():
                out.append((idx, mask.copy()))
            self._hand_edited.discard((oid, idx))
            self._anchored.add((oid, idx))
        return out

    def _sam_targets(self, worms):
        """Which worms a tracking run should cover.

        Kept on this side rather than read from the worker's backend: that
        object belongs to the other thread."""
        known = set(self._prompted_worms) | {oid for (oid, _f) in self._sam_points}
        if worms is None:
            worms = sorted(known)
            for oid, _f in sorted(self._hand_edited):
                if oid not in worms:
                    worms.append(oid)
        return [oid for oid in worms
                if oid in known or any(w == oid for (w, _f) in self._hand_edited)]

    def _sam_propagate(self, start=None, also_backwards=False, worms=None):
        """Queue a tracking run and return immediately.

        Each worm is tracked in its own session, one after another, so they
        cannot overwrite each other. `start=None` covers the whole clip;
        `start=f` re-tracks from frame f, optionally backwards as well.

        The GUI stays live throughout: masks land frame by frame as they arrive,
        and anything you edit meanwhile is protected from the results."""
        if not self._ensure_segmenter():
            return
        worms = self._sam_targets(worms)
        if not worms:
            self.info.setText("Add SAM points on a worm first "
                              "(or correct a mask and press “Mask → prompt”).")
            return

        runs = [(start, False)] + ([(start, True)] if also_backwards
                                   and start is not None else [])
        per_worm = sum(self.n if s is None else (s + 1 if rev else self.n - s)
                       for s, rev in runs)
        total = per_worm * len(worms)
        plan = [(oid, self._pending_anchors(oid), runs) for oid in worms]
        self._prompted_worms.update(worms)

        self._tracked = set()
        self._kept_locked = set()            # results dropped to keep your masks
        self._edited_during_run = set()      # your edits win over these results
        self._track_started = (start, also_backwards, len(worms))
        if self.sam.busy:
            self.info.setText("SAM2 is busy — this run is queued.")
        self.progress_panel.start("Segmenting", total,
                                  f"{len(worms)} worm(s) — you can keep editing")
        self.sam.track(plan, total)

    def _finish_tracking(self, frames_done):
        """A tracking run ended (or was stopped): recompute and redraw once."""
        self.progress_panel.finish()
        start, also_backwards, _n = getattr(self, "_track_started",
                                            (None, False, 0))
        for oid in self._tracked:
            if oid not in self.by_id:
                self._insert_worm(Worm(obj_id=oid))
        for oid in self._tracked:
            w = self.by_id[oid]
            for idx in self.masks.frames_of(oid):
                self._apply_mask_stats(w, idx)
            self._recompute_kinematics(w)
        if self._tracked and self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()
        self.plot.set_worm(self.selected)
        self.plot.set_frame(self.frame)
        self._refresh_sel_skel()
        self._render_overlay()
        self._update_info()
        kept = len(self._kept_locked)
        where = "whole clip" if start is None else f"frame {start} onwards"
        print(f"[sam2] tracked {frames_done} frames ({where}) "
              f"-> worms {sorted(self._tracked)}"
              + (f"; kept {kept} corrected frame(s)" if kept else ""))
        if kept:
            self.info.setText(f"Tracking done.\n{kept} corrected frame(s) were "
                              f"kept as you left them.")
        self._edited_during_run = set()
        self._kept_locked = set()

    # --- clearing ------------------------------------------------------------
    def _sam_clear(self):
        """Forget every prompt. The masks on screen stay — only SAM2's sessions
        are dropped, so previously anchored frames go back to being plain hand
        edits that will be re-anchored on the next run."""
        self._sam_points.clear()
        self._hand_edited |= self._anchored
        self._anchored.clear()
        if self.sam is not None:
            self.sam.stop_current()
            self.sam.reset()
        self._render_overlay()
