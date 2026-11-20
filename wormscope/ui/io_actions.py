"""File-level actions: open / import a dataset, calibrate units, save, export."""

import shutil
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

from .. import settings
from ..core.metrics import plot_specs, unit_scales
from ..io.export import export_overlay
from ..io.loaders import build_raw_dataset, has_saved_work, load_dataset
from ..io.manual_lengths import MANUAL_FILE, write_manual_lengths
from ..io.session import (SCALE_FILE, ensure_frames_in, save_scale_image,
                          write_session)
from ..io.video import SOURCE_FILE, VideoCancelled, is_video, video_info
from ..io.writers import save_measurements_plot, write_measurements_csv
from .calibration_dialog import CalibrationDialog
from .file_dialogs import pick_folder, pick_video_or_image
from .progress import Progress
from .video_import_dialog import VideoImportDialog


def save_as_target(picked: Path, current: Path) -> Path:
    """Where Save as… puts the video when the user picks `picked`: that folder
    itself when it is empty (or is the current one), otherwise a subfolder
    named after the video — '<name>', or '<name>_2', '_3', … when taken — so
    picking a folder full of other things never mixes files into it."""
    if picked.resolve() == current.resolve() or not any(picked.iterdir()):
        return picked
    candidate, n = picked / current.name, 2
    while candidate.exists() and (has_saved_work(candidate)
                                  or any(candidate.iterdir())):
        if candidate.resolve() == current.resolve():
            return candidate
        candidate = picked / f"{current.name}_{n}"
        n += 1
    return candidate


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
        d = pick_video_or_image(
            self, "Select a video file, an image, or a folder of frames", start)
        if not d:
            return
        if is_video(d):
            ds = self._decode_video(Path(d))
            if ds is False:                  # stopped or failed: already said
                return
        else:
            ds = build_raw_dataset(Path(d))
        if ds is None:
            QMessageBox.warning(self, "Upload video or image",
                                f"No image files found in:\n{d}")
            return
        self._open_dataset(ds)

    def _decode_video(self, video):
        """Upload a video file: ask which frames to keep (all, a lower frame
        rate, or random ones), decode them (with progress and a Stop button),
        then open them. False when cancelled, stopped or failed."""
        try:
            info = video_info(video)
        except ImportError:
            QMessageBox.warning(self, "Upload video",
                                "Reading video files needs imageio with ffmpeg:"
                                "\n\n  pip install imageio imageio-ffmpeg")
            return False
        except Exception as err:             # noqa: BLE001 - shown to the user
            self._video_error(video, err)
            return False
        dlg = VideoImportDialog(self, video, info)
        if dlg.exec() != VideoImportDialog.DialogCode.Accepted:
            return False
        sampling = dlg.sampling()
        bar = Progress(self, "Reading video", 0,
                       label=f"Decoding {video.name} into frames…")

        def progress(done, total):
            if total and total != bar.total:
                bar.set_total(total)         # the estimate from the file header
            return bar.step(done, f"{video.name}: {done}/{total} frames")

        try:
            ds = build_raw_dataset(video, progress=progress, sampling=sampling)
        except VideoCancelled:
            bar.close()
            print(f"[video] decoding {video.name} stopped")
            return False
        except ImportError:
            bar.close()
            QMessageBox.warning(self, "Upload video",
                                "Reading video files needs imageio with ffmpeg:"
                                "\n\n  pip install imageio imageio-ffmpeg")
            return False
        except Exception as err:             # noqa: BLE001 - shown to the user
            bar.close()
            self._video_error(video, err)
            return False
        bar.close()
        return ds

    def _video_error(self, video, err):
        # ffmpeg's own report is pages long: its first and last lines say it
        lines = [x for x in str(err).splitlines() if x.strip()]
        why = lines[0] if lines else type(err).__name__
        if len(lines) > 1 and "stderr" in str(err):
            why += f"\n{lines[-1]}"
        print(f"[video] could not read {video}: {err}")
        QMessageBox.warning(self, "Upload video",
                            f"Could not read the video\n{video}\n\n{why}")

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
            state=self._calibration_state(), frame=frame,
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
        self._refresh_measure_view()        # manual length in the new units

    def _calibration_state(self):
        """What the Units… dialog opens with. A scale.png in the video folder
        is loaded automatically when no other scale image is set (or the one
        that was set has gone)."""
        state = dict(self._calibration)
        img = state.get("image")
        if img and Path(img).is_file():
            return state
        if state.get("source") == "frame" and not img:
            return state                     # calibrated on a video frame
        scale = self._data_dir / SCALE_FILE if self._data_dir else None
        if scale is not None and scale.is_file():
            state["image"] = str(scale)
            state["source"] = "file"
            if img and Path(img).resolve() != scale.resolve():
                state.pop("line", None)      # drawn on another image
        return state

    # --- saving --------------------------------------------------------------
    def _save(self):
        """Write everything “Continue editing video…” needs to pick up again:
        dirty masks + skeletons (removed files deleted), the frames themselves
        if they live elsewhere, measurements.csv + plot, manual_lengths.csv,
        the scale image as scale.png, and wormscope.json with the settings
        (calibration, lock edits, corrected frames)."""
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
        n_manual = write_manual_lengths(folder / MANUAL_FILE, self.worms,
                                        self.ds.idx_to_key, self.px_to_um)
        if n_manual or (folder / MANUAL_FILE).exists():
            print(f"[save] {n_manual} manual lengths -> {MANUAL_FILE}")
        self._save_scale_image(folder)
        write_session(folder, self._session_state())
        self._settings_dirty = False
        self._manual_dirty = False
        self.save_btn.setEnabled(False)
        self._update_title()
        print("[save] masks, skeletons and measurements.csv written")

    # --- Save as…: move the video to another folder --------------------------
    def _save_folder(self):
        """The folder Save writes to (None for the demo / no video)."""
        return self.ds.csv_path.parent if self.ds.csv_path else None

    def _save_as(self):
        """Save the whole video — frames, masks, skeletons, measurements,
        manual lengths, scale image, settings — into another folder, and keep
        working there. The old folder is left exactly as it was."""
        if not self.masks.savable or not self.n:
            return
        current = self._save_folder()
        start = settings.get("save_dir") or str(current.parent)
        picked = pick_folder(self, "Save video to folder", start)
        if not picked:
            return
        dest = save_as_target(Path(picked), current)
        if dest.resolve() == current.resolve():
            self._save()
            return
        try:
            self._move_to_folder(dest)
            self._save()
        except Exception as err:             # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Save as",
                                f"Could not save to\n{dest}\n\n{err}")
            return
        settings.save({"save_dir": str(dest.parent)})
        QMessageBox.information(self, "Save as",
                                f"Saved to:\n{dest}\n\nFrom now on Save "
                                f"writes here. The previous folder is "
                                f"unchanged:\n{current}")

    def _move_to_folder(self, dest):
        """Point everything at `dest`: frames linked / copied in, masks and
        skeletons copied, every later Save written there."""
        dest.mkdir(parents=True, exist_ok=True)
        added = ensure_frames_in(dest, self._frames_dir)
        self.frames.rebase(dest / "frames")
        copied = self.masks.move_to(dest / "binary", dest / "skel")
        src = self._save_folder() / SOURCE_FILE   # where the frames came from
        if src.is_file():
            shutil.copy2(src, dest / SOURCE_FILE)
        print(f"[save as] {dest}: {added} frames, {copied} masks copied")
        self.ds.csv_path = dest / "measurements.csv"
        self.ds.frames_dir = self._frames_dir = dest / "frames"
        self._data_dir = dest
        self.ds.title = self._base_title = f"wormscope — {dest.name}"
        self._pending_delete.clear()         # those files are in the old folder
        self._settings_dirty = True
        self._manual_dirty = True
        self._update_save_folder()

    def _update_save_folder(self):
        folder = self._save_folder()
        if folder is None or not self.masks.savable:
            self.save_folder_label.setText("not saved (no folder)")
            self.save_folder_label.setToolTip("")
            return
        self.save_folder_label.setText(f"→ {folder.name}")
        self.save_folder_label.setToolTip(f"Save writes to:\n{folder}\n\n"
                                          f"Save as… (⌘⇧S / Ctrl+Shift+S) "
                                          f"picks another folder.")
        self._update_title()

    def _save_scale_image(self, folder):
        """Copy the scale image used in Units… into the folder as scale.png
        and point the calibration at that copy (the original path is kept as
        "original_image")."""
        src = self._calibration.get("image")
        if not src:
            return
        dst = save_scale_image(folder, src)
        if dst is None:
            return
        if Path(src).resolve() != dst.resolve():
            # A scale.png carried over from another video folder (Save as…)
            # keeps the original it was made from.
            if not (Path(src).name == SCALE_FILE
                    and self._calibration.get("original_image")):
                self._calibration["original_image"] = str(src)
            print(f"[save] scale image -> {SCALE_FILE}")
        self._calibration["image"] = str(dst)

    # --- session settings (wormscope.json) ----------------------------------
    def _session_state(self):
        """The settings that make reopening feel like you never left."""
        corrected = {}
        for oid, idx in sorted(self._hand_edited | self._anchored):
            if oid in self.by_id:
                corrected.setdefault(str(oid), []).append(
                    int(self.ds.idx_to_key.get(idx, idx)))
        dialog = dict(self._calibration)
        folder = self.ds.csv_path.parent if self.ds.csv_path else None
        img = dialog.get("image")
        if img and folder is not None and Path(img).parent.resolve() \
                == folder.resolve():
            dialog["image"] = Path(img).name          # relative: folder can move
        return {"frames": "frames",
                "calibration": {"um_per_px": self.px_to_um, "fps": self.fps,
                                "dialog": dialog},
                "lock_edits": self.lock_box.isChecked(),
                "annotation_mode": self.annotation_mode,
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
        self._calibration = dict(cal.get("dialog") or {})
        img = self._calibration.get("image")
        if img and not Path(img).is_absolute() and self._data_dir is not None:
            self._calibration["image"] = str(self._data_dir / img)
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
        return (bool(self.masks.dirty) or self._settings_dirty
                or self._manual_dirty)

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
