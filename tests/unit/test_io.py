"""Reading a data folder and writing results back out.

The alignment rule is the thing to protect: original frames, per-worm masks and
CSV rows are matched on the *trailing integer* in each filename, so
``frame_0042.png``, ``worm01_frame_0042.png`` and CSV row ``42`` are the same
frame even though none of the names look alike.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
from PIL import Image

from wormscope.config import ACCEL_METRIC, SPEED_METRIC
from wormscope.core.models import MaskStore, Worm
from wormscope.io.export import export_frame
from wormscope.io.loaders import (build_raw_dataset, find_frames_dir,
                                  load_dataset)
from wormscope.io.writers import write_measurements_csv


# --- loading a pipeline folder ---------------------------------------------
def test_load_dataset_finds_frames_masks_and_metrics(data_folder):
    ds = load_dataset(data_folder)
    assert ds is not None and len(ds.frames) == 4
    assert [w.obj_id for w in ds.worms] == [1]
    assert ds.masks.savable, "masks loaded from disk can be saved back"
    assert ds.csv_path == data_folder / "measurements.csv"


def test_masks_line_up_with_their_frames(data_folder):
    ds = load_dataset(data_folder)
    worm = ds.worms[0]
    for f in range(4):
        assert ds.masks.has_mask(1, f)
        assert worm.centroid(f) is not None
    x0 = worm.centroid(0)[0]
    x3 = worm.centroid(3)[0]
    assert x3 - x0 == pytest.approx(3.0), \
        "the mask drifts 1 px per frame — if the frames were mismatched this " \
        "would not be monotone"


def test_metrics_come_from_the_csv_when_there_is_one(data_folder):
    ds = load_dataset(data_folder)
    worm = ds.worms[0]
    assert worm.metrics["area_px"][2] == 42, "row 2 of the CSV, not the mask"


def test_without_a_csv_the_metrics_are_measured_from_the_masks(data_folder):
    (data_folder / "measurements.csv").unlink()
    ds = load_dataset(data_folder)
    worm = ds.worms[0]
    assert worm.metrics["area_px"][0] == 40, "10 x 4 pixels, counted"


def test_a_folder_with_no_frames_loads_as_nothing(tmp_path):
    assert load_dataset(tmp_path) is None


def test_find_frames_dir_accepts_the_usual_layouts(tmp_path):
    root = tmp_path / "vid"
    (root / "frames").mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(root / "frames" / "a_1.png")
    assert find_frames_dir(root) == root / "frames"


# --- importing raw frames ---------------------------------------------------
def test_raw_frames_become_an_empty_working_set(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for f in range(3):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(raw / f"img_{f}.png")
    ds = build_raw_dataset(raw)
    assert len(ds.frames) == 3 and ds.worms == [], \
        "nothing is segmented yet — that is what the SAM2 tools are for"
    assert ds.frames_dir == raw


def test_importing_never_writes_into_the_raw_folder(tmp_path):
    """The originals are somebody's microscope data. Outputs go to a sibling."""
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(raw / "img_0.png")
    ds = build_raw_dataset(raw)
    assert ds.masks.binary_dir is not None
    assert raw not in ds.masks.binary_dir.parents and \
        ds.masks.binary_dir.parent.name == "raw_wormscope"


def test_a_folder_with_no_images_imports_as_nothing(tmp_path):
    assert build_raw_dataset(tmp_path) is None


# --- writing measurements.csv ----------------------------------------------
def worm_with_metrics(wid=1):
    w = Worm(obj_id=wid)
    w.metrics["area_px"] = {0: 100.0, 1: 110.0}
    w.metrics["length_px"] = {0: 20.0, 1: 21.0}
    w.metrics[SPEED_METRIC] = {1: 2.0}
    w.metrics[ACCEL_METRIC] = {}
    return w


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_the_csv_always_has_the_pixel_columns(tmp_path):
    out = tmp_path / "measurements.csv"
    n = write_measurements_csv(out, [worm_with_metrics()], {0: 0, 1: 1},
                               "area_px", "length_px")
    rows = read_csv(out)
    assert n == 2 and len(rows) == 2
    assert list(rows[0]) == ["frame", "worm_id", "area_px", "length_px",
                             "speed_px_per_frame", "accel_px_per_frame2"]
    assert rows[0]["area_px"] == "100" and rows[1]["speed_px_per_frame"] == "2.0"


def test_a_missing_value_is_written_as_an_empty_cell(tmp_path):
    """Not 0 — 'we did not measure this' and 'it measured zero' are different
    things to whoever analyses the CSV afterwards."""
    out = tmp_path / "m.csv"
    write_measurements_csv(out, [worm_with_metrics()], {0: 0, 1: 1},
                           "area_px", "length_px")
    assert read_csv(out)[0]["speed_px_per_frame"] == ""


def test_rows_are_written_under_the_original_frame_numbers(tmp_path):
    """Display index 0 may be frame 137 in the source data; the CSV has to
    speak the source's numbering."""
    out = tmp_path / "m.csv"
    write_measurements_csv(out, [worm_with_metrics()], {0: 137, 1: 138},
                           "area_px", "length_px")
    assert [r["frame"] for r in read_csv(out)] == ["137", "138"]


def test_calibrated_columns_are_appended_not_substituted(tmp_path):
    out = tmp_path / "m.csv"
    write_measurements_csv(out, [worm_with_metrics()], {0: 0, 1: 1},
                           "area_px", "length_px", px_to_um=2.0, fps=10.0)
    row = read_csv(out)[0]
    assert row["area_px"] == "100", "the pixel measurement is still there"
    assert float(row["area_um2"]) == pytest.approx(400.0)
    assert "speed_um_per_s" in row and "speed_um_per_frame" not in row


def test_without_fps_the_calibrated_speed_is_per_frame(tmp_path):
    out = tmp_path / "m.csv"
    write_measurements_csv(out, [worm_with_metrics()], {0: 0, 1: 1},
                           "area_px", "length_px", px_to_um=2.0)
    row = read_csv(out)[1]
    assert float(row["speed_um_per_frame"]) == pytest.approx(4.0)


def test_rows_are_sorted_by_frame_then_worm(tmp_path):
    out = tmp_path / "m.csv"
    write_measurements_csv(out, [worm_with_metrics(5), worm_with_metrics(2)],
                           {0: 0, 1: 1}, "area_px", "length_px")
    rows = read_csv(out)
    assert [(r["frame"], r["worm_id"]) for r in rows] == \
        [("0", "2"), ("0", "5"), ("1", "2"), ("1", "5")]


def test_writing_to_no_path_is_a_no_op():
    """The demo has no csv_path; Save must not invent a file in the cwd."""
    assert write_measurements_csv(None, [worm_with_metrics()], {}, "a", "l") == 0


def test_the_csv_reflects_edits_not_the_file_it_was_loaded_from(data_folder):
    """The whole point of the program: correct a mask, and the numbers that
    leave are the corrected ones."""
    from wormscope.core.metrics import apply_mask_stats
    ds = load_dataset(data_folder)
    worm = ds.worms[0]
    ds.masks.set(1, 0, np.zeros((30, 40), bool))
    apply_mask_stats(worm, ds.masks, 0, "area_px", "length_px")
    out = data_folder / "measurements.csv"
    write_measurements_csv(out, ds.worms, ds.idx_to_key, "area_px", "length_px")
    assert read_csv(out)[0]["area_px"] == "0"


# --- exporting an overlay frame --------------------------------------------
def test_export_frame_draws_the_current_masks(dataset):
    out = export_frame(dataset.frames, dataset.worms, dataset.masks, 0)
    raw = dataset.frames.get(0)
    assert out.shape == raw.shape and out.dtype == np.uint8
    assert not np.array_equal(out, raw), "the worm is drawn on top"
    assert np.array_equal(out[0, 0], raw[0, 0]), "empty background is untouched"


def test_export_frame_does_not_modify_the_cached_frame(dataset):
    before = dataset.frames.get(0).copy()
    export_frame(dataset.frames, dataset.worms, dataset.masks, 0)
    assert np.array_equal(dataset.frames.get(0), before), \
        "the frame cache is shared with the display — exporting must copy"


def test_export_frame_of_an_absent_worm_is_just_the_frame():
    from wormscope.core.models import Dataset, FrameSource
    frames = FrameSource([np.full((10, 10, 3), 50, np.uint8)], lambda a: a)
    worm = Worm(obj_id=1)                     # no bbox anywhere
    out = export_frame(frames, [worm], MaskStore((10, 10)), 0)
    assert np.array_equal(out, frames.get(0))
