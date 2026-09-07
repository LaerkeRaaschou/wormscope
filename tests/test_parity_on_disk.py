"""Against the original, on disk: load a real folder, edit, Save, compare bytes.

The other parity test works in memory. This one takes the whole round trip —
read PNGs and a CSV off disk, paint, press Save — and compares the files that
land, byte for byte. It is the check that the refactor did not change what
somebody's data folder looks like afterwards.

Skipped when ``wormscope_gui3.py`` is not around; see ``WORMSCOPE_ORIGINAL``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.needs_original


def build_fake_dataset(root, n=6):
    """frames/ + binary/ + skel/ + measurements.csv, worm ids 1 and 2."""
    (root / "frames").mkdir(parents=True)
    (root / "binary").mkdir()
    (root / "skel").mkdir()
    rng = np.random.default_rng(0)
    rows = ["frame,worm_id,area_px,length_px"]
    for f in range(n):
        img = rng.integers(80, 180, size=(60, 90, 3)).astype(np.uint8)
        Image.fromarray(img).save(root / "frames" / f"img_{f:03d}.png")
        for wid in (1, 2):
            m = np.zeros((60, 90), bool)
            y = 12 + 20 * (wid - 1) + f % 3
            m[y:y + 6, 10 + 2 * f:34 + 2 * f] = True
            Image.fromarray(np.where(m, 255, 0).astype(np.uint8), "L").save(
                root / "binary" / f"worm{wid:02d}_img_{f:03d}.png")
            rows.append(f"{f},{wid},{int(m.sum())},{24.0}")
    (root / "measurements.csv").write_text("\n".join(rows) + "\n")


def drive(load_dataset, MainWindow, data_dir):
    """Open the folder, paint one stroke, Save — then read back what is there."""
    ds = load_dataset(data_dir)
    win = MainWindow(ds)
    out = {"n": win.n,
           "ids": [w.obj_id for w in win.worms],
           "metric_names": list(ds.metric_names),
           "idx_to_key": dict(ds.idx_to_key)}
    win.set_frame(2)
    win.select_worm(win.worms[0])
    win.zoomv.mode = win.zoomv.MODE_DRAW
    win.zoomv.brush = 4
    win._on_paint(50.0, 20.0)
    win._on_paint(58.0, 26.0)
    win._on_stroke_finished()
    out["skel_len"] = int(np.count_nonzero(win.masks.skel(win.worms[0].obj_id, 2)))
    win._save()
    win.close()

    out["files"] = {str(p.relative_to(data_dir)): p.read_bytes()
                    for p in sorted(data_dir.rglob("*"))
                    if p.is_file() and p.suffix in (".png", ".csv")}
    return out


@pytest.fixture(scope="module")
def both_on_disk(qapp, tmp_path_factory):
    path = None
    for candidate in (os.environ.get("WORMSCOPE_ORIGINAL"),
                      "wormscope_gui3.py",
                      "/mnt/user-data/uploads/wormscope/wormscope_gui3.py"):
        if candidate and os.path.isfile(candidate):
            path = candidate
            break
    if path is None:
        pytest.skip("wormscope_gui3.py not found")

    tmp = tmp_path_factory.mktemp("roundtrip")
    src = tmp / "src"
    build_fake_dataset(src)
    old_dir, new_dir = tmp / "old", tmp / "new"
    shutil.copytree(src, old_dir)
    shutil.copytree(src, new_dir)

    spec = importlib.util.spec_from_file_location("orig_gui_disk", path)
    orig = importlib.util.module_from_spec(spec)
    sys.modules["orig_gui_disk"] = orig
    spec.loader.exec_module(orig)
    old = drive(orig.load_dataset, orig.MainWindow, old_dir)

    from wormscope.io.loaders import load_dataset
    from wormscope.ui import MainWindow
    new = drive(load_dataset, MainWindow, new_dir)
    return old, new


@pytest.mark.parametrize("key", ["n", "ids", "metric_names", "idx_to_key",
                                 "skel_len"])
def test_the_folder_is_read_the_same_way(both_on_disk, key):
    old, new = both_on_disk
    assert old[key] == new[key]


def test_the_same_files_are_written(both_on_disk):
    old, new = both_on_disk
    assert sorted(old["files"]) == sorted(new["files"])
    assert len(old["files"]) > 10, "a Save that wrote nothing proves nothing"


def test_every_file_is_byte_identical(both_on_disk):
    old, new = both_on_disk
    differing = [k for k in old["files"]
                 if old["files"][k] != new["files"].get(k)]
    assert not differing, f"these files differ: {differing}"
