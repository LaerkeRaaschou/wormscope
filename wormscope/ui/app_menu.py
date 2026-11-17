"""The menu bar: SAM2 model choice, desktop shortcut and updates.

These are the things a user sets once (or rarely), so they live in a menu
rather than taking room next to the editing tools.
"""

import datetime
import os
import threading

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QApplication, QMessageBox

from .. import __version__, settings, shortcuts, updates
from ..segmentation import sam2_models


class _UpdateNotifier(QObject):
    """Carries the result of the background version check to the GUI thread
    (a signal emitted from another thread is delivered in the receiver's)."""
    found = pyqtSignal(str)


class AppMenuMixin:

    def _show_save_folder(self):
        """Open the folder Save writes to in Finder / Explorer."""
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        folder = self._save_folder()
        if folder is not None and folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _build_menu(self):
        bar = self.menuBar()

        file_menu = bar.addMenu("File")
        # Shortcuts are window-wide QShortcuts (layout.py); shown here as text.
        file_menu.addAction("Save\t⌘S / Ctrl+S", self._save)
        file_menu.addAction("Save as…\t⌘⇧S / Ctrl+Shift+S", self._save_as)
        file_menu.addAction("Show save folder", self._show_save_folder)

        sam = bar.addMenu("SAM2")
        size = sam.addMenu("Model size")
        self._model_actions = {}
        group = QActionGroup(self)
        for key, (label, _cfg, _ckpt, mb) in sam2_models.MODELS.items():
            act = QAction(f"{label} (~{mb} MB)", self, checkable=True)
            act.triggered.connect(lambda _c=False, k=key: self._pick_sam2_model(k))
            group.addAction(act)
            size.addAction(act)
            self._model_actions[key] = act
        size.aboutToShow.connect(self._refresh_model_menu)
        self._refresh_model_menu()
        sam.addAction("Download the model now", self._download_sam2_model)
        sam.addSeparator()
        sam.addAction("Choose files manually…",
                      lambda: self._choose_sam2_files(explain=False))

        help_menu = bar.addMenu("Help")
        help_menu.addAction("Create desktop shortcut", self._make_shortcut)
        help_menu.addAction("Check for updates…",
                            lambda: self._check_updates(quiet=False))
        self._auto_update_action = QAction("Check for updates at startup", self,
                                           checkable=True)
        self._auto_update_action.setChecked(
            settings.get("check_updates", "on") != "off")
        self._auto_update_action.toggled.connect(
            lambda on: settings.save({"check_updates": "on" if on else "off"}))
        help_menu.addAction(self._auto_update_action)
        help_menu.addSeparator()
        help_menu.addAction("About wormscope", self._about)

    # --- SAM2 model ----------------------------------------------------------
    def _manual_sam2_files(self):
        saved = settings.load()
        return bool(saved.get("sam2_cfg") or saved.get("sam2_ckpt"))

    def _refresh_model_menu(self):
        """Tick the current size; mark which ones are already downloaded. With
        hand-picked files in use, no size is ticked."""
        manual = self._manual_sam2_files()
        current = sam2_models.current_model()
        for key, act in self._model_actions.items():
            label, _cfg, _ckpt, mb = sam2_models.MODELS[key]
            have = sam2_models.find_checkpoint(key) is not None
            act.setText(f"{label} " + ("(downloaded)" if have else f"(~{mb} MB)"))
            act.setChecked(not manual and key == current)

    def _pick_sam2_model(self, key):
        sam2_models.set_model(key)
        self._shutdown_sam()                 # the next SAM click loads it
        label = sam2_models.MODELS[key][0]
        if sam2_models.find_checkpoint(key) is None:
            if self._offer_sam2_download():
                self.info.setText(f"SAM2 model: {label}.")
        else:
            self.info.setText(f"SAM2 model: {label}.")
            self._start_sam(warm=True)

    # --- shortcut ------------------------------------------------------------
    def _make_shortcut(self):
        try:
            where = shortcuts.create_desktop_shortcut()
        except Exception as err:             # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Desktop shortcut",
                                f"Could not create the shortcut:\n\n{err}")
            return
        QMessageBox.information(self, "Desktop shortcut",
                                f"Created:\n{where}\n\nDouble-click it to start "
                                f"wormscope.")

    # --- updates -------------------------------------------------------------
    def _start_update_check(self):
        """Look for a newer version in the background, at most once a day, and
        only say something if there is one."""
        if settings.get("check_updates", "on") == "off":
            return
        today = datetime.date.today().isoformat()
        if settings.get("last_update_check") == today:
            return
        settings.save({"last_update_check": today})
        self._update_notifier = _UpdateNotifier()
        self._update_notifier.found.connect(self._on_update_found)

        def look():
            try:
                remote = updates.latest_version()
            except Exception as err:         # noqa: BLE001 - offline is fine
                print(f"[update] could not check: {err}")
                return
            if updates.is_newer(remote):
                self._update_notifier.found.emit(remote)

        threading.Thread(target=look, daemon=True).start()

    def _check_updates(self, quiet=False):
        try:
            remote = updates.latest_version()
        except Exception as err:             # noqa: BLE001 - shown to the user
            if not quiet:
                QMessageBox.warning(self, "Updates",
                                    f"Could not check for updates:\n\n{err}")
            return
        if updates.is_newer(remote):
            self._on_update_found(remote)
        elif not quiet:
            QMessageBox.information(self, "Updates",
                                    f"wormscope {__version__} is up to date.")

    def _on_update_found(self, remote):
        text = (f"wormscope {remote} is available (you have {__version__}).")
        if not updates.can_self_update():
            QMessageBox.information(self, "Update available",
                                    f"{text}\n\n{updates.how_to_update()}")
            return
        if os.name == "nt":
            # Windows locks the files of a running program: close, then update.
            if QMessageBox.question(
                    self, "Update available",
                    f"{text}\n\nUpdate now? Save your work first — wormscope "
                    f"will close, the update runs in a separate window, and "
                    f"then you can start wormscope again.") \
                    != QMessageBox.StandardButton.Yes:
                return
            try:
                updates.update_after_exit()
            except Exception as err:         # noqa: BLE001 - shown to the user
                QMessageBox.warning(self, "Update failed", str(err))
                return
            QApplication.quit()
            return
        if QMessageBox.question(self, "Update available",
                                f"{text}\n\nUpdate now? This can take a few "
                                f"minutes; restart wormscope afterwards.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.info.setText("Updating wormscope… (the window may not respond)")
        try:
            ok, output = updates.self_update()
        except Exception as err:             # noqa: BLE001 - shown to the user
            ok, output = False, str(err)
        if ok:
            QMessageBox.information(self, "Updated",
                                    f"wormscope was updated to {remote}.\n\n"
                                    f"Close and reopen it to use the new version.")
        else:
            QMessageBox.warning(self, "Update failed",
                                f"The update did not finish:\n\n{output[-1500:]}")

    def _about(self):
        cfg, ckpt, source = settings.sam2_paths()
        QMessageBox.information(
            self, "About wormscope",
            f"wormscope {__version__}\n\n"
            f"SAM2 config: {cfg or '(none)'}\nSAM2 checkpoint: {ckpt or '(none)'}"
            f"\n({source})\n\nSettings: {settings.settings_path()}")
