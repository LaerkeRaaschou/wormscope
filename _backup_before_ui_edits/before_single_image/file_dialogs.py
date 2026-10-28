"""File and folder pickers used everywhere wormscope opens something.

On a Mac these use Qt's own picker instead of the macOS one: the macOS
picker (NSOpenPanel) can crash the whole app with "trace trap" when it is
opened from a Python.org Python + PyQt6 app, most reliably from inside
another dialog (Units… -> "Load scale image…"). Windows and Linux keep
their normal system picker.
"""

import sys

from PyQt6.QtWidgets import QFileDialog

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
