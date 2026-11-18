"""
Opens the GUI with the dataset (if called)
"""
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .core.geometry import HAVE_SKIMAGE
from .core.models import empty_dataset
from .io.loaders import build_raw_dataset, is_photo, load_dataset
from .io.video import is_video
from .theme import apply_theme
from .ui import WINDOWS, MainWindow


def resolve_dataset(path=None):
    """ Find data_dir from argument, or open empty """
    if path is None:
        print("[start] no folder given — open one with “Upload video or image…” "
              "or “Continue editing video…”")
        return empty_dataset()

    path = Path(path).expanduser()
    if is_photo(path):                       # a single image: a one-frame video
        return build_raw_dataset(path) or empty_dataset()
    if is_video(path):                       # a video file: decoded into frames
        def report(done, total):
            if done % 100 == 0:
                print(f"[video] {done}/{total or '?'} frames")
            return False
        try:
            return build_raw_dataset(path, progress=report) or empty_dataset()
        except Exception as err:             # noqa: BLE001 - start empty instead
            print(f"[warn] could not read the video {path}: {err}")
            return empty_dataset()
    if not path.is_dir():
        print(f"[warn] {path} is not a folder, an image or a video; starting empty")
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


def _log_to_file_if_windowless():
    """Started from a shortcut / the windowed launcher there is no console, so
    print() output and tracebacks would vanish. Send them to a log file instead,
    so there is something to look at when a user reports a problem."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    log = Path.home() / ".wormscope" / "wormscope.log"
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = open(log, "a", buffering=1, encoding="utf-8")
    except OSError:
        return
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


def main():
    """Entry point of the `wormscope` command (see pyproject.toml):
    ``wormscope`` or ``wormscope path/to/folder``."""
    _log_to_file_if_windowless()
    path = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run(path))
