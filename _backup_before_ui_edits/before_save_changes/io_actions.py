"""File-level actions: open / import a dataset, calibrate units, save, export."""

from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..core.metrics import plot_specs, unit_scales
from ..io.export import export_overlay
from ..io.loaders import build_raw_dataset, load_dataset
from ..io.writers import save_measurements_plot, write_measurements_csv
from .calibration_dialog import CalibrationDialog
from .progress import Progress


class IOActionsMixin:

    # --- opening another dataset --------------------------------------------
    def _confirm_discard(self, what):
        """True when it is safe to throw the current edits away."""
        if not self.masks.dirty:
            return True
        return QMessageBox.question(
            self, "Unsaved edits",
            f"You have unsaved edits. Discard them and {what}?",
            QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel) \
            == QMessageBox.StandardButton.Discard

    def _open_folder(self):
        if not self._confirm_discard("open another video"):
            return
        start = str(self._data_dir.parent) if self._data_dir else str(Path.home())
        d = QFileDialog.getExistingDirectory(
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
        if not self._confirm_discard("import raw frames"):
            return
        start = str(self._frames_dir.parent) if self._frames_dir else str(Path.home())
        d = QFileDialog.getExistingDirectory(
            self, "Select a folder of raw frames", start)
        if not d:
            return
        ds = build_raw_dataset(Path(d))
        if ds is None:
            QMessageBox.warning(self, "Upload new video frames",
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
        self.plot.set_scales(self._unit_scales())
        self.plot.set_frame(self.frame)
        self._update_info()

    # --- saving --------------------------------------------------------------
    def _save(self):
        """Write dirty masks + skeletons, delete removed files, rewrite the CSV
        and regenerate measurements_plot.png."""
        if not self.masks.savable:
            return
        for wid, idx in list(self.masks.dirty):
            self.masks.save(wid, idx)
        for p in list(self._pending_delete):            # deleted / renamed files
            try:
                Path(p).unlink(missing_ok=True)
            except OSError as err:
                print(f"[save] could not delete {p}: {err}")
        self._pending_delete.clear()
        self._write_csv()
        if save_measurements_plot(self.ds.csv_path.parent / "measurements_plot.png",
                                  self.worms, self._plot_specs()):
            print("[save] regenerated measurements_plot.png")
        self.save_btn.setEnabled(False)
        self._update_title()
        print("[save] masks, skeletons and measurements.csv written")

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
