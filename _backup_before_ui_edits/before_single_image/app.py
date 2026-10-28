"""
Opens the GUI with the dataset (if called)
"""
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .core.geometry import HAVE_SKIMAGE
from .core.models import empty_dataset
from .io.loaders import build_raw_dataset, load_dataset
from .theme import apply_theme
from .ui import WINDOWS, MainWindow


def resolve_dataset(path=None):
    """ Find data_dir from argument, or open empty """
    if path is None:
        print("[start] no folder given — open one with “Upload new video frames…” "
              "or “Continue editing video…”")
        return empty_dataset()

    path = Path(path).expanduser()
    if not path.is_dir():
        print(f"[warn] {path} is not a folder; starting empty")
        return empty_dataset()

    ds = load_dataset(path) or build_raw_dataset(path)
    if ds is None:
        print(f"[warn] no frames found in {path}; starting empty")
        return empty_dataset()
    return ds


def run(path=None, argv=None):
    """ Start the GUI """
    ds = resolve_dataset(path)

    if not HAVE_SKIMAGE:
        print("[warn] scikit-image not installed: length/skeleton recompute "
              "disabled (pip install scikit-image)")

    app = QApplication(argv if argv is not None else sys.argv)
    apply_theme(app)
    win = MainWindow(ds)
    WINDOWS.append(win)
    win.show()
    return app.exec()
