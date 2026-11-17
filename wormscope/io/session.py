"""What a saved video folder holds besides the masks: its own frames and a
small ``wormscope.json`` with the session settings.

After Save, a video folder is self-contained — exactly what
“Continue editing video…” needs to pick up where you left off::

    <video>/
        frames/              # the original frames (linked or copied in on Save)
        binary/              # worm{id:02d}_{stem}.png  per-worm masks
        skel/                # worm{id:02d}_{stem}.png  skeletons
        measurements.csv     # per-frame metrics
        measurements_plot.png
        manual_lengths.csv   # manual length measurements (when there are any)
        scale.png            # the scale image used to calibrate (Units…)
        wormscope.json       # calibration, lock edits, frames you corrected

Frames are hard-linked when the source is on the same disk (no extra space,
instant) and copied otherwise. Frames already inside the folder are left alone.
"""

import json
import os
import shutil
from pathlib import Path

from ..core.utils import list_images, sort_by_frame_number

SESSION_FILE = "wormscope.json"
SESSION_VERSION = 1
SCALE_FILE = "scale.png"


def read_session(folder) -> dict:
    """The saved session settings, or {} when there are none (or they are
    unreadable — a broken settings file must never stop a video opening)."""
    p = Path(folder) / SESSION_FILE
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as err:
        print(f"[session] ignoring unreadable {p.name}: {err}")
        return {}
    return data if isinstance(data, dict) else {}


def write_session(folder, state: dict):
    """Write wormscope.json via a temporary file, so a crash mid-write can't
    leave a half-written file behind."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / (SESSION_FILE + ".tmp")
    tmp.write_text(json.dumps({"version": SESSION_VERSION, **state}, indent=2))
    os.replace(tmp, folder / SESSION_FILE)


def save_scale_image(folder, source) -> Path | None:
    """Put the calibration's scale image in the video folder as scale.png, so
    the folder keeps the image its µm/pixel was measured on. Converted to PNG
    when it is another format; left alone when `source` already is that file.
    Returns the path of scale.png, or None when `source` can't be read."""
    folder = Path(folder)
    dst = folder / SCALE_FILE
    source = Path(source)
    if not source.is_file():
        return dst if dst.is_file() else None
    if dst.exists() and source.resolve() == dst.resolve():
        return dst
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / (SCALE_FILE + ".tmp.png")
    try:
        from PIL import Image
        with Image.open(source) as im:
            im.save(tmp, format="PNG")       # keeps 16-bit grey where PNG can
    except Exception:                        # noqa: BLE001 - try the loader
        try:
            from PIL import Image

            from .loaders import load_image_rgb
            Image.fromarray(load_image_rgb(source)).save(tmp, format="PNG")
        except Exception as err:             # noqa: BLE001
            print(f"[session] could not save {source} as {SCALE_FILE}: {err}")
            tmp.unlink(missing_ok=True)
            return dst if dst.is_file() else None
    os.replace(tmp, dst)
    return dst


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def ensure_frames_in(folder, frames_dir) -> int:
    """Make sure `folder`/frames holds the video's frames. `frames_dir` is a
    folder of frames, or a single image file (a photo: a one-frame video).
    Returns how many files were added (0 when they were already there)."""
    folder = Path(folder)
    if frames_dir is None or _inside(Path(frames_dir), folder):
        return 0
    source = Path(frames_dir)
    sources = [source] if source.is_file() else sort_by_frame_number(list_images(source))
    dest = folder / "frames"
    dest.mkdir(parents=True, exist_ok=True)
    added = 0
    for src in sources:
        target = dest / src.name
        if target.exists():
            continue
        try:
            os.link(src, target)             # same disk: no copy needed
        except OSError:
            shutil.copy2(src, target)
        added += 1
    return added
