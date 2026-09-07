"""Writing results back out: measurements.csv and measurements_plot.png."""

import csv

import numpy as np

from ..config import ACCEL_METRIC, SPEED_METRIC
from ..core.utils import color_for


def write_measurements_csv(csv_path, worms, idx_to_key, area_metric,
                           length_metric, px_to_um=None, fps=None):
    """Rewrite measurements.csv from the current (edited) metrics.

    Always writes the pixel columns; when `px_to_um` is set, physical-unit
    columns are appended (per second when `fps` is known, else per frame).
    Returns the number of rows written."""
    if csv_path is None:
        return 0

    def num(v, nd):
        return "" if v is None or v != v else round(float(v), nd)

    cal = bool(px_to_um)
    u = px_to_um or 1.0
    fps = fps or None
    fields = ["frame", "worm_id", "area_px", "length_px",
              "speed_px_per_frame", "accel_px_per_frame2"]
    if cal:
        fields += ["area_um2", "length_um"]
        fields += ["speed_um_per_s", "accel_um_per_s2"] if fps \
            else ["speed_um_per_frame", "accel_um_per_frame2"]

    rows = []
    for w in worms:
        idxs = set().union(*(w.metrics.get(m, {}) for m in
                             (area_metric, length_metric,
                              SPEED_METRIC, ACCEL_METRIC))) or set()
        for idx in sorted(idxs):
            area = w.metrics.get(area_metric, {}).get(idx)
            length = w.metrics.get(length_metric, {}).get(idx)
            speed = w.metrics.get(SPEED_METRIC, {}).get(idx)
            accel = w.metrics.get(ACCEL_METRIC, {}).get(idx)
            row = {
                "frame": idx_to_key.get(idx, idx),
                "worm_id": w.obj_id,
                "area_px": "" if area is None else int(area),
                "length_px": num(length, 3),
                "speed_px_per_frame": num(speed, 4),
                "accel_px_per_frame2": num(accel, 4),
            }
            if cal:
                row["area_um2"] = num(None if area is None else area * u * u, 4)
                row["length_um"] = num(None if length is None else length * u, 4)
                sf = u * fps if fps else u
                af = u * fps * fps if fps else u
                sk = "speed_um_per_s" if fps else "speed_um_per_frame"
                ak = "accel_um_per_s2" if fps else "accel_um_per_frame2"
                row[sk] = num(None if speed is None else speed * sf, 5)
                row[ak] = num(None if accel is None else accel * af, 5)
            rows.append(row)
    rows.sort(key=lambda r: (r["frame"], r["worm_id"]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def save_measurements_plot(path, worms, specs):
    """Regenerate the per-video overview plot: one subplot per metric spec
    (name, ylabel, factor), one coloured line per worm. Returns True on success.

    Uses the Agg backend directly so it works regardless of the Qt binding."""
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except Exception as err:                      # pragma: no cover
        print(f"[plot] matplotlib unavailable: {err}")
        return False
    fig = Figure(figsize=(11, 2.3 * len(specs) + 1))
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(len(specs), 1, sharex=True))
    any_line = False
    for ax, (name, label, factor) in zip(axes, specs):
        for w in worms:
            series = w.metrics.get(name, {})
            fs = [f for f in sorted(series) if series[f] == series[f]]
            if not fs:
                continue
            any_line = True
            ax.plot(fs, [series[f] * factor for f in fs],
                    color=color_for(w.obj_id) / 255.0, lw=1, marker=".", ms=3,
                    label=f"worm {w.obj_id}")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame")
    if any_line:
        axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                       fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    return True
