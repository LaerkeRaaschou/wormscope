"""File and folder pickers used everywhere wormscope opens something.

``pick_video_or_image`` is the one exception to the rule below: no system
picker can choose "a folder OR a file" in one window, so it always uses Qt's
own picker, on every platform.

On a Mac these use Qt's own picker instead of the macOS one: the macOS
picker (NSOpenPanel) can crash the whole app with "trace trap" when it is
opened from a Python.org Python + PyQt6 app, most reliably from inside
another dialog (Units… -> "Load scale image…"). Windows and Linux keep
their normal system picker.
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import QDialog, QFileDialog, QGridLayout, QPushButton

ON_MAC = sys.platform == "darwin"
OPTIONS = (QFileDialog.Option.DontUseNativeDialog if ON_MAC
           else QFileDialog.Option(0))


def pick_folder(parent, title, start=""):
    """A folder path, or "" when cancelled."""
    return QFileDialog.getExistingDirectory(
        parent, title, start,
        OPTIONS | QFileDialog.Option.ShowDirsOnly)


def pick_file(parent, title, start="", file_filter=""):
    """A file path, or "" when cancelled."""
    path, _ = QFileDialog.getOpenFileName(parent, title, start, file_filter,
                                          options=OPTIONS)
    return path


PHOTO_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff)"


def pick_video_or_image(parent, title, start=""):
    """A folder of frames (a video) or a single image (a photo).

    Pick an image and press Open for a photo. For a video, go into the folder
    — or just click it — and press “Open folder as video”. Returns the path,
    or "" when cancelled."""
    dialog = QFileDialog(parent, title, start, PHOTO_FILTER)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog)  # needs the extra button
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)

    chosen = {"folder": None, "highlighted": None}
    dialog.currentChanged.connect(lambda p: chosen.update(highlighted=p))

    def open_folder():
        # the folder clicked on, otherwise the folder being looked at
        h = chosen["highlighted"]
        chosen["folder"] = h if h and Path(h).is_dir() else dialog.directory().absolutePath()
        # QDialog's accept, not QFileDialog's: that one would open the folder
        # (or refuse, with no file chosen) instead of closing the picker
        QDialog.accept(dialog)

    button = QPushButton("Open folder as video")
    button.setToolTip("Use every image in the selected folder as the frames of one video")
    button.clicked.connect(open_folder)
    layout = dialog.layout()
    if isinstance(layout, QGridLayout):       # Qt's own picker: a row under the buttons
        layout.addWidget(button, layout.rowCount(), 0, 1, layout.columnCount())

    accepted = dialog.exec()
    files = dialog.selectedFiles()
    dialog.deleteLater()
    if not accepted:
        return ""
    if chosen["folder"]:
        return chosen["folder"]
    return files[0] if files else ""
