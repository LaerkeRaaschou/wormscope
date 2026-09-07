"""Shared fixtures: a headless Qt app, small deterministic datasets, a stub
SAM2 backend, and the golden-file machinery.

Two things happen at import time, before anything else can go wrong:

* Qt is forced onto the ``offscreen`` platform, so the suite runs with no
  display and never opens a window on your screen;
* ``WORMSCOPE_SETTINGS`` is redirected into a temporary file, so running the
  tests can never read or overwrite your real ``~/.wormscope/settings.json``.

Both are set with ``setdefault``/an explicit temp path rather than assumed, so
running one test file directly behaves exactly like running the whole suite.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# --- process-wide setup, before importing Qt or wormscope -------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_SETTINGS_TMP = Path(tempfile.mkdtemp(prefix="wormscope-tests-"))
os.environ["WORMSCOPE_SETTINGS"] = str(_SETTINGS_TMP / "settings.json")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                        # noqa: E402
import pytest                                             # noqa: E402

GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"


# --- command line -----------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens", action="store_true", default=False,
        help="rewrite the recorded outputs in tests/goldens/ instead of "
             "comparing against them (use after a change you meant to make)")


def pytest_collection_modifyitems(config, items):
    """Skip the tests that need the original single-file GUI when it isn't
    there — most checkouts won't have it, and that is fine."""
    if _original_gui_path() is None:
        skip = pytest.mark.skip(reason="wormscope_gui3.py not found "
                                       "(set WORMSCOPE_ORIGINAL to point at it)")
        for item in items:
            if "needs_original" in item.keywords:
                item.add_marker(skip)


def _original_gui_path():
    """The pre-refactor single-file GUI, if it is anywhere obvious."""
    env = os.environ.get("WORMSCOPE_ORIGINAL")
    candidates = [Path(env)] if env else []
    candidates += [
        ROOT / "wormscope_gui3.py",
        ROOT.parent / "wormscope_gui3.py",
        Path("/mnt/user-data/uploads/wormscope/wormscope_gui3.py"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


@pytest.fixture(scope="session")
def original_gui():
    """The original single-file GUI, imported as a module."""
    import importlib.util
    path = _original_gui_path()
    assert path is not None
    spec = importlib.util.spec_from_file_location("orig_gui", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orig_gui"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Qt ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole session — Qt allows no more than one."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture
def silence_dialogs(monkeypatch):
    """Answer every modal dialog automatically.

    A test that pops a real QMessageBox would hang forever waiting for a click
    that is never coming, so anything touching Delete, Save-over or the SAM2
    file picker takes this fixture. Returns a list of (kind, title) so a test
    can assert that the user *was* asked."""
    from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
    asked = []

    def record(kind, answer):
        def fake(*args, **kwargs):
            asked.append((kind, args[1] if len(args) > 1 else ""))
            return answer
        return fake

    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(record("information", None)))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(record("warning", None)))
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(record("critical", None)))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(record("question",
                                            QMessageBox.StandardButton.Yes)))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(record("open", ("", ""))))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(record("dir", "")))
    monkeypatch.setattr(QInputDialog, "getInt",
                        staticmethod(record("int", (0, False))))
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(record("double", (0.0, False))))
    return asked


# --- data -------------------------------------------------------------------
def make_mask(shape, box):
    """A rectangular mask — the simplest thing with a known area and bbox."""
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def build_dataset(n_frames=6, shape=(40, 60), worm_ids=(1,), frames_dir=None,
                  with_masks=True, csv_path=None):
    """A tiny in-memory Dataset: flat grey frames and one rectangle per worm.

    Deliberately boring — every number a test asserts on (area 80, centroid at
    the rectangle's middle) can be worked out by hand from the box below."""
    from wormscope.core.metrics import apply_mask_stats
    from wormscope.core.models import Dataset, FrameSource, MaskStore, Worm

    H, W = shape
    frames = FrameSource([np.full((H, W, 3), 170, np.uint8)
                          for _ in range(n_frames)], lambda a: a)
    masks = MaskStore(shape)
    worms = []
    for n, wid in enumerate(worm_ids):
        w = Worm(obj_id=wid)
        if with_masks:
            for f in range(n_frames):
                # drifts one pixel per frame, so speed is exactly 1 px/frame
                x = 5 + f + 15 * n
                masks.set(wid, f, make_mask(shape, (x, 8, x + 20, 12)))
                apply_mask_stats(w, masks, f, "area_px", "length_px")
        worms.append(w)
    masks.dirty.clear()
    for w in worms:
        from wormscope.core.kinematics import refresh_kinematics
        refresh_kinematics(w)
    return Dataset(frames, worms, ["area_px", "length_px"], masks,
                   {i: i for i in range(n_frames)}, csv_path, "test",
                   frames_dir=frames_dir)


@pytest.fixture
def dataset():
    """One worm, six frames, masks already measured."""
    return build_dataset()


@pytest.fixture
def window(qapp, dataset):
    """A MainWindow on the tiny dataset. Closed for you afterwards."""
    from wormscope.ui import MainWindow
    win = MainWindow(dataset)
    win.zoomv.resize(240, 180)
    yield win
    win._shutdown_sam()
    win.close()


@pytest.fixture
def make_window(qapp):
    """Factory for windows on datasets a test builds itself."""
    from wormscope.ui import MainWindow
    made = []

    def _make(ds):
        win = MainWindow(ds)
        win.zoomv.resize(240, 180)
        made.append(win)
        return win
    yield _make
    for win in made:
        win._shutdown_sam()
        win.close()


@pytest.fixture
def data_folder(tmp_path):
    """A folder in the on-disk layout loaders.load_dataset expects:

        <video>/frames/frame_0000.png …
        <video>/binary/worm01_frame_0000.png …
        <video>/measurements.csv
    """
    from PIL import Image
    root = tmp_path / "video"
    (root / "frames").mkdir(parents=True)
    (root / "binary").mkdir()
    (root / "skel").mkdir()
    for f in range(4):
        Image.fromarray(np.full((30, 40, 3), 160, np.uint8)).save(
            root / "frames" / f"frame_{f:04d}.png")
        m = np.zeros((30, 40), np.uint8)
        m[10:14, 5 + f:15 + f] = 255
        Image.fromarray(m, "L").save(root / "binary" / f"worm01_frame_{f:04d}.png")
    (root / "measurements.csv").write_text(
        "frame,worm_id,area_px,length_px\n"
        + "".join(f"{f},1,{40 + f},{9.0 + f}\n" for f in range(4)))
    return root


# --- SAM2 stand-in ----------------------------------------------------------
class StubBackend:
    """Stands in for Sam2Backend: records every call, returns simple masks.

    This is the contract a segmentation backend has to satisfy — if you swap
    SAM2 for another model, making this class's methods work is the whole job.
    `slow` adds a per-frame delay so the asynchronous behaviour is real."""

    slow = 0.0

    def __init__(self, frames, shape):
        self.shape = shape
        self.n_frames = len(frames)
        self.calls = []
        self.point_prompts = []
        self.sessions = {}               # obj_id -> its prompts, as SAM2 would
        self._prompted = set()

    def session(self, obj_id):
        return self.sessions.setdefault(int(obj_id), [])

    def forget(self, obj_id):
        self.sessions.pop(int(obj_id), None)
        self._prompted.discard(int(obj_id))

    def prompted_objects(self):
        return [o for o in self.sessions if o in self._prompted]

    def has_prompts(self, obj_id):
        return int(obj_id) in self._prompted

    def add_points(self, frame_idx, obj_id, points, labels):
        self.calls.append(("add_points", frame_idx, obj_id, list(points),
                           list(labels)))
        self.session(obj_id).append(("points", frame_idx))
        self._prompted.add(int(obj_id))
        self.point_prompts = list(zip(points, labels))
        m = np.zeros(self.shape, bool)
        m[5:9, 5:20] = True
        return m

    def add_mask(self, frame_idx, obj_id, mask):
        self.calls.append(("add_mask", frame_idx, obj_id,
                           int(np.asarray(mask).sum())))
        self.session(obj_id).append(("mask", frame_idx))
        self._prompted.add(int(obj_id))
        return np.asarray(mask, bool)

    def propagate(self, obj_id, start_frame_idx=None, reverse=False):
        self.calls.append(("propagate", obj_id, start_frame_idx, reverse))
        start = 0 if start_frame_idx is None else start_frame_idx
        order = range(start, -1, -1) if reverse else range(start, self.n_frames)
        for f in order:
            if self.slow:
                time.sleep(self.slow)
            m = np.zeros(self.shape, bool)
            m[10:14, f:f + 10] = True
            yield f, m

    def prewarm(self, indices, progress=None):
        self.calls.append(("prewarm", list(indices)))
        return len(indices)

    def estimate_prewarm_bytes(self, n_frames=None):
        return 1_000_000

    def reset(self):
        self.calls.append(("reset",))
        self.sessions.clear()
        self._prompted.clear()


@pytest.fixture
def stub_sam(monkeypatch):
    """Install StubBackend as the segmentation backend for this test."""
    from wormscope import segmentation
    monkeypatch.setattr(segmentation, "SEGMENTER_FACTORY", StubBackend)
    monkeypatch.setattr(StubBackend, "slow", 0.0)
    return StubBackend


@pytest.fixture
def settle(qapp):
    """Pump the Qt loop until the SAM2 worker is idle and its signals delivered.

    Everything SAM2 does happens on another thread, so a test that just calls
    ``win._sam_propagate()`` and asserts would be checking an empty room."""
    def _settle(win, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            qapp.processEvents()
            if win.sam is None or not win.sam.busy:
                for _ in range(3):           # let queued signals arrive
                    qapp.processEvents()
                    time.sleep(0.02)
                if win.sam is None or not win.sam.busy:
                    return True
            time.sleep(0.01)
        raise AssertionError("the SAM2 worker never went idle")
    return _settle


# --- goldens ----------------------------------------------------------------
@pytest.fixture
def golden(request):
    """Compare a value against a recorded copy in tests/goldens/.

        golden("demo_metrics", value)

    First run (or ``pytest --update-goldens``) records it; every run after that
    compares. Floats are rounded before storing, so a golden survives the last
    bit of arithmetic noise but not a real change of formula."""
    update = request.config.getoption("--update-goldens")

    def _round(obj, nd=6):
        if isinstance(obj, dict):
            return {str(k): _round(v, nd) for k, v in sorted(obj.items(),
                                                             key=lambda kv: str(kv[0]))}
        if isinstance(obj, (list, tuple)):
            return [_round(v, nd) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return None if obj != obj else round(float(obj), nd)
        if isinstance(obj, (np.integer, int, bool)):
            return int(obj)
        return obj

    def _check(name, value, nd=6):
        path = GOLDEN_DIR / f"{name}.json"
        payload = _round(value, nd)
        if update or not path.exists():
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
            if not update:
                pytest.skip(f"recorded a new golden: {path.name}")
            return
        recorded = json.loads(path.read_text())
        assert payload == recorded, (
            f"{name} no longer matches tests/goldens/{path.name}.\n"
            f"If you meant to change this, re-record with: "
            f"pytest --update-goldens")
    return _check
