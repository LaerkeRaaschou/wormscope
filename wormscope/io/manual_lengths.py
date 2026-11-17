"""manual_lengths.csv: the manual length measurements of one video.

One row per worm and frame that was measured by hand — frames nobody
annotated simply have no row. Compare with the automatic length by joining on
(frame, worm_id) with measurements.csv::

    frame,worm_id,length_px,length_um,n_points,points
    12,1,412.71,165.08,7,"[[103.0, 88.5], [140.25, 95.0], ...]"

``frame`` is the original frame number (as in measurements.csv), ``points``
the clicked points in image pixels, in order along the worm. ``length_um`` is
empty until the video is calibrated (Units…); it is recomputed from the
points on every Save, so recalibrating never leaves it stale. The points are
what is read back: the lengths are always derived from them.
"""

import csv
import json
from pathlib import Path

from ..core.manual import polyline_length

MANUAL_FILE = "manual_lengths.csv"
FIELDS = ["frame", "worm_id", "length_px", "length_um", "n_points", "points"]


def write_manual_lengths(path, worms, idx_to_key, px_to_um=None) -> int:
    """Rewrite manual_lengths.csv from the worms' manual measurements.
    Measurements with fewer than two points are skipped. When there is
    nothing to write and no file yet, no file is created. Returns the number
    of rows written."""
    path = Path(path)
    rows = []
    for w in worms:
        for idx, pts in getattr(w, "manual", {}).items():
            if len(pts) < 2:
                continue
            length = polyline_length(pts)
            rows.append({
                "frame": idx_to_key.get(idx, idx),
                "worm_id": w.obj_id,
                "length_px": round(length, 3),
                "length_um": round(length * px_to_um, 4) if px_to_um else "",
                "n_points": len(pts),
                "points": json.dumps([[round(float(x), 2), round(float(y), 2)]
                                      for x, y in pts]),
            })
    if not rows and not path.exists():
        return 0
    rows.sort(key=lambda r: (r["frame"], r["worm_id"]))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return len(rows)


def read_manual_lengths(path, key_to_idx, n_frames=None) -> dict:
    """{worm_id: {frame index: [(x, y), ...]}} from manual_lengths.csv, or {}
    when there is none. Rows that can't be read, or whose frame isn't in this
    video, are skipped with a message rather than stopping the video opening."""
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict = {}
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error) as err:
        print(f"[manual] ignoring unreadable {path.name}: {err}")
        return {}
    skipped = 0
    for r in rows:
        try:
            key = int(float(r["frame"]))
            wid = int(float(r["worm_id"]))
            pts = [(float(x), float(y)) for x, y in json.loads(r["points"])]
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        idx = key_to_idx.get(key)
        if idx is None and n_frames is not None and 0 <= key < n_frames \
                and not key_to_idx:
            idx = key
        if idx is None or len(pts) < 2:
            skipped += 1
            continue
        out.setdefault(wid, {})[idx] = pts
    n = sum(len(v) for v in out.values())
    print(f"[manual] {n} manual lengths for worms {sorted(out)}"
          + (f" ({skipped} rows skipped)" if skipped else ""))
    return out
