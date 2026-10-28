"""File-level actions: open / import a dataset, calibrate units, save, export."""

from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

from ..core.metrics import plot_specs, unit_scales
from ..io.export import export_overlay
from ..io.loaders import build_raw_dataset, load_dataset
from ..io.session import ensure_frames_in, write_session
from ..io.writers import save_measurements_plot, write_measurements_csv
from .calibration_dialog import CalibrationDialog
from .file_dialogs import pick_folder, pick_video_or_image
from .progress import Progress


class IOActionsMixin:

    # --- opening another dataset --------------------------------------------
    def _confirm_discard(self, what):
        """True when it is safe to go on and do `what` (e.g. "close wormscope"):
        nothing is unsaved, or the user chose Save (and it worked) or Don't
        Save. False on Cancel.

        Used before opening another video and before closing the window."""
        if not self._has_unsaved():
            return True
        Btn = QMessageBox.StandardButton
        buttons = Btn.Discard | Btn.Cancel
        if self.masks.savable:               # the demo data has nowhere to save to
            buttons |= Btn.Save
        box = QMessageBox(QMessageBox.Icon.Warning, "Unsaved edits",
                          f"You have unsaved edits. Save them before you {what}?",
                          buttons, self)
        box.setDefaultButton(Btn.Save if self.masks.savable else Btn.Cancel)
        box.setEscapeButton(Btn.Cancel)
        answer = box.exec()
        if answer == Btn.Discard:
            return True
        if answer != Btn.Save:
            return False
        try:
            self._save()
        except Exception as err:             # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Save failed",
                                f"Your edits could not be saved:\n\n{err}")
            return False
        return not self._has_unsaved()

    def _open_folder(self):
        if not self._confirm_discard("open another video"):
            return
        start = str(self._data_dir.parent) if self._data_dir else str(Path.home())
        d = pick_folder(
            self, "Select a video folder (containing frames/ and binary/)", start)
        if not d:
            return
        ds = load_dataset(Path(d))
        if ds is None:
            QMessageBox.warning(
                self, "Continue editing video",
                f"No frames found in:\n{d}\n\nExpected a folder that contains a "
                f"frames/ subfolder (and usually binary/).")
            return
        self._open_dataset(ds)

    def _import_raw(self):
        """Upload a folder of frames (a video) or a single image (a photo)."""
        if not self._confirm_discard("upload a new video or image"):
            return
        # next to the current video's _wormscope folder, where its raw frames are
        start = str(self._data_dir.parent) if self._data_dir else str(Path.home())
        d = pick_video_or_image(self, "Select an image, or a folder of frames", start)
        if not d:
            return
        ds = build_raw_dataset(Path(d))
        if ds is None:
            QMessageBox.warning(self, "Upload video or image",
                                f"No image files found in:\n{d}")
            return
        self._open_dataset(ds)

    def _open_dataset(self, ds):
        """Show a fresh window for `ds` and close this one."""
        from .main_window import MainWindow, WINDOWS      # local: avoids a cycle
        win = MainWindow(ds)
        win.resize(self.size())
        win.move(self.pos())
        win.show()
        WINDOWS.append(win)
        if self in WINDOWS:
            WINDOWS.remove(self)
        self._closing_confirmed = True       # already asked about unsaved edits
        self.close()

    # --- physical units ------------------------------------------------------
    def _unit_scales(self):
        """metric name -> (factor, label). Empty (px) until calibrated."""
        return unit_scales(self.area_metric, self.length_metric,
                           self.px_to_um, self.fps)

    def _plot_specs(self):
        """(name, label, factor) for the four metrics, honouring calibration."""
        return plot_specs(self.area_metric, self.length_metric,
                          self.px_to_um, self.fps)

    def _set_units(self):
        """Units…: the calibration dialog (scale image + line, and fps)."""
        start = (self._data_dir or (self._frames_dir.parent if self._frames_dir
                                    else None))
        frame = self.frames.get(self.frame) if self.n else None
        dlg = CalibrationDialog(
            self, px_to_um=self.px_to_um, fps=self.fps,
            state=self._calibration, frame=frame,
            frame_shape=frame.shape[:2] if frame is not None else None,
            start_dir=str(start) if start else None)
        if dlg.exec() != CalibrationDialog.DialogCode.Accepted:
            return
        self._calibration = dlg.state()
        self.px_to_um = dlg.px_to_um()
        self.fps = dlg.fps()
        self._settings_changed()             # saved with the video on Save
        self.plot.set_scales(self._unit_scales())
        self.plot.set_frame(self.frame)
        self._update_info()

    # --- saving --------------------------------------------------------------
    def _save(self):
        """Write everything “Continue editing video…” needs to pick up again:
        dirty masks + skeletons (removed files deleted), the frames themselves
        if they live elsewhere, measurements.csv + plot, and wormscope.json
        with the settings (calibration, lock edits, corrected frames)."""
        if not self.masks.savable:
            return
        folder = self.ds.csv_path.parent
        added = ensure_frames_in(folder, self._frames_dir)
        if added:
            print(f"[save] put {added} frames in {folder.name}/frames/")
        for wid, idx in list(self.masks.dirty):
            self.masks.save(wid, idx)
        for p in list(self._pending_delete):            # deleted / renamed files
            try:
                Path(p).unlink(missing_ok=True)
            except OSError as err:
                print(f"[save] could not delete {p}: {err}")
        self._pending_delete.clear()
        self._write_csv()
        if save_measurements_plot(folder / "measurements_plot.png",
                                  self.worms, self._plot_specs()):
            print("[save] regenerated measurements_plot.png")
        write_session(folder, self._session_state())
        self._settings_dirty = False
        self.save_btn.setEnabled(False)
        self._update_title()
        print("[save] masks, skeletons and measurements.csv written")

    # --- session settings (wormscope.json) ----------------------------------
    def _session_state(self):
        """The settings that make reopening feel like you never left."""
        corrected = {}
        for oid, idx in sorted(self._hand_edited | self._anchored):
            if oid in self.by_id:
                corrected.setdefault(str(oid), []).append(
                    int(self.ds.idx_to_key.get(idx, idx)))
        return {"frames": "frames",
                "calibration": {"um_per_px": self.px_to_um, "fps": self.fps,
                                "dialog": self._calibration},
                "lock_edits": self.lock_box.isChecked(),
                "corrected_frames": corrected}

    def _apply_session(self, state):
        """Restore what _session_state saved. Corrected frames come back as
        hand edits: the new SAM2 session has never seen them, so they are
        sent again as mask prompts on the next run — and stay protected by
        “lock edits”."""
        if not state:
            return
        cal = state.get("calibration") or {}
        self.px_to_um = cal.get("um_per_px") or None
        self.fps = cal.get("fps") or None
        self._calibration = cal.get("dialog") or {}
        self.lock_box.blockSignals(True)
        self.lock_box.setChecked(bool(state.get("lock_edits")))
        self.lock_box.blockSignals(False)
        key_to_idx = {k: i for i, k in self.ds.idx_to_key.items()}
        for oid, keys in (state.get("corrected_frames") or {}).items():
            for k in keys:
                idx = key_to_idx.get(k)
                if idx is not None and int(oid) in self.by_id:
                    self._hand_edited.add((int(oid), idx))
        self.plot.set_scales(self._unit_scales())

    def _settings_changed(self, *_):
        """A setting that is saved with the video changed: offer Save."""
        if self.masks.savable:
            self._settings_dirty = True
            self.save_btn.setEnabled(True)
        self._update_title()

    def _has_unsaved(self):
        return bool(self.masks.dirty) or self._settings_dirty

    def _write_csv(self):
        write_measurements_csv(self.ds.csv_path, self.worms, self.ds.idx_to_key,
                               self.area_metric, self.length_metric,
                               self.px_to_um, self.fps)

    # --- export corrected overlay video -------------------------------------
    def _export_video(self):
        out_dir = self.ds.csv_path.parent if self.ds.csv_path else Path.cwd()
        bar = Progress(self, "Export overlay", self.n,
                       label=f"Rendering {self.n} frames…")
        try:
            target, wrote_video, written = export_overlay(
                self.frames, self.worms, self.masks, self.n, out_dir,
                idx_to_key=self.ds.idx_to_key, fps=self.fps,
                progress=lambda done, total: bar.step(
                    done, f"Rendered {done}/{total} frames"))
        except ImportError:
            bar.close()
            QMessageBox.warning(self, "Export video",
                                "This needs the imageio package:\n\n"
                                "  pip install imageio")
            return
        except Exception as err:
            bar.close()
            QMessageBox.warning(self, "Export video", f"Export failed:\n\n{err}")
            return
        bar.close()
        stopped = written < self.n
        head = (f"Stopped after {written} of {self.n} frames"
                if stopped else f"Saved {written} frames")
        if wrote_video:
            msg = f"{head} to:\n{target}"
        else:
            msg = (f"ffmpeg not available, so {head.lower()} as PNGs in:\n"
                   f"{target}\n\nFor an .mp4, install it with:\n"
                   f"  pip install imageio-ffmpeg")
        print(f"[export] wrote {written}/{self.n} frames -> {target}")
        QMessageBox.information(self, "Export video", msg)
