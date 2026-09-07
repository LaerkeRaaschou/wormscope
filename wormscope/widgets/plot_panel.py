"""The Plots panel: one stacked subplot per metric, scrubbable.

Falls back to a plain label when matplotlib isn't installed, so the rest of the
GUI keeps working. Both variants expose the same five setters
(``set_worm``, ``set_frame``, ``set_smooth``, ``set_scales``,
``set_normalized``) plus the ``frameSelected`` signal.

Metrics named in ``set_normalized`` are drawn divided by their own median, so
area and length become "how much bigger / smaller than usual is the worm right
now" — a shape you can read at a glance, and one that survives an arbitrary
change of units. The real number is never lost: hover the curve and it appears
in a small box.
"""

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel

from ..config import (DARK_BLUE, GRIDCOL, PANEL, PLOT_AX_BG, PLOT_DOT,
                      PLOT_LINE, SMOOTH_WINDOW)
from ..core.utils import moving_average

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False


if HAVE_MPL:
    class PlotPanel(FigureCanvasQTAgg):
        frameSelected = pyqtSignal(int)

        def __init__(self, metrics):
            self.metrics = metrics or []
            self.worm = None
            self._frame = 0
            self.smooth = False
            self.scales = {}                 # metric -> (factor, ylabel)
            self.normalized = set()          # metrics drawn ÷ their median
            self._drawn = []                 # per-axis {frame: plotted-y}
            self._raw = []                   # per-axis {frame: real value}
            self._units = []                 # per-axis unit label for the tip
            self._tips = []                  # per-axis hover annotation
            self._dragging = False
            self.fig = Figure(figsize=(3, 3), tight_layout=True)
            self.fig.patch.set_facecolor(PANEL)
            super().__init__(self.fig)
            n = max(1, len(self.metrics))
            self.axes = np.atleast_1d(self.fig.subplots(n, 1, sharex=True))
            self.dots = [None] * len(self.axes)
            self._blank()
            self.mpl_connect("button_press_event", self._on_press)
            self.mpl_connect("motion_notify_event", self._on_motion)
            self.mpl_connect("button_release_event", self._on_release)
            self.mpl_connect("figure_leave_event", self._on_leave)

        def _blank(self):
            for ax, name in zip(self.axes, self.metrics):
                ax.clear()
                ax.set_facecolor(PLOT_AX_BG)
                ax.set_ylabel(name, fontsize=8, color=DARK_BLUE)
                ax.tick_params(labelsize=7, colors=DARK_BLUE)
                for sp in ax.spines.values():
                    sp.set_color(GRIDCOL)
                ax.grid(alpha=0.35, color=GRIDCOL)
            if not self.metrics:
                self.axes[0].clear()
                self.axes[0].set_facecolor(PLOT_AX_BG)
            if self.worm is None:
                self.axes[0].text(0.5, 0.5, "select a worm", ha="center",
                                  va="center", transform=self.axes[0].transAxes,
                                  fontsize=9, color=GRIDCOL)
            elif self.metrics:
                self.axes[-1].set_xlabel("frame", fontsize=8, color=DARK_BLUE)
            self.dots = [None] * len(self.axes)
            self._tips = [None] * len(self.axes)   # ax.clear() dropped them
            self.draw_idle()

        def set_smooth(self, on):
            self.smooth = bool(on)
            self.set_worm(self.worm)         # redraw with/without smoothing

        def set_scales(self, scales):
            self.scales = scales or {}
            self.set_worm(self.worm)         # redraw in the new units

        def set_normalized(self, names):
            """Draw these metrics divided by their own median.

            Area and length are the ones worth normalizing: their absolute
            value depends on the worm and on the calibration, so the useful
            question is not "how many pixels" but "how far from this worm's
            own normal". Everything else (speed, acceleration) keeps its
            units, where zero and sign carry the meaning."""
            self.normalized = set(names or ())
            self.set_worm(self.worm)

        def set_worm(self, worm):
            self.worm = worm
            self._blank()
            self._drawn = [{} for _ in self.axes]
            self._raw = [{} for _ in self.axes]
            self._units = [""] * len(self.axes)
            if worm is None or not self.metrics:
                return
            for i, (ax, name) in enumerate(zip(self.axes, self.metrics)):
                factor, label = self.scales.get(name, (1.0, name))
                ax.set_ylabel(label, fontsize=8, color=DARK_BLUE)
                self._units[i] = label
                series = worm.metrics.get(name, {})
                fs = [f for f in sorted(series) if series[f] == series[f]]
                if not fs:
                    ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color=GRIDCOL)
                    continue
                real = [series[f] * factor for f in fs]
                self._raw[i] = dict(zip(fs, real))    # what the tooltip reports
                ys = real
                if name in self.normalized:
                    med = float(np.median(real))
                    # A zero or non-finite median means there is nothing to
                    # divide by (an all-empty worm); leave the curve alone
                    # rather than turning it into infinities.
                    if med == med and med != 0:
                        ys = [v / med for v in real]
                        ax.set_ylabel(f"{label} / median", fontsize=8,
                                      color=DARK_BLUE)
                        ax.axhline(1.0, color=GRIDCOL, lw=0.8, ls="--",
                                   alpha=0.7, zorder=1)
                if self.smooth:
                    ys = moving_average(ys, SMOOTH_WINDOW)
                ax.plot(fs, ys, color=PLOT_LINE, lw=1.3)
                self._drawn[i] = dict(zip(fs, ys))    # what the dot should track
            self._place_dots()
            self.draw_idle()

        def set_frame(self, f):
            self._frame = f
            if self.worm is None or not self.metrics:
                return
            self._place_dots()
            self.draw_idle()

        def _place_dots(self):
            f = self._frame
            for i, ax in enumerate(self.axes):
                val = self._drawn[i].get(f) if i < len(self._drawn) else None
                if self.dots[i] is None:
                    (self.dots[i],) = ax.plot([], [], "o", color=PLOT_DOT,
                                              markersize=6, markeredgecolor="white",
                                              markeredgewidth=0.8, zorder=5)
                self.dots[i].set_data([] if val is None else [f],
                                      [] if val is None else [val])

        def _emit_frame(self, event):
            if event.inaxes is not None and event.xdata is not None:
                self.frameSelected.emit(int(round(event.xdata)))

        def _on_press(self, event):
            if event.button == 1 and event.inaxes is not None:
                self._dragging = True
                self._emit_frame(event)      # jump to the clicked frame

        def _on_motion(self, event):
            if self._dragging:               # scrub while the button is held
                self._emit_frame(event)
                return
            self._hover(event)

        def _on_release(self, event):
            self._dragging = False

        def _on_leave(self, event):
            self._hide_tips()

        # --- hover read-out ---------------------------------------------------
        @staticmethod
        def _fmt(v):
            """Enough digits to be useful, few enough to read in a small box."""
            a = abs(v)
            if a and (a < 0.01 or a >= 1e5):
                return f"{v:.3g}"
            return f"{v:.1f}" if a >= 10 else f"{v:.2f}"

        def _tip(self, i):
            if self._tips[i] is None:
                tip = self.axes[i].annotate(
                    "", xy=(0, 0), xytext=(8, 8), textcoords="offset points",
                    fontsize=7, color=DARK_BLUE, zorder=6, annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.28", fc=PANEL, ec=GRIDCOL,
                              lw=0.6, alpha=0.95))
                # Kept out of the layout calculation: the figure uses
                # tight_layout, which would otherwise squeeze the axes sideways
                # to make room for the box — the plot would shuffle about under
                # the cursor every time the read-out appeared.
                tip.set_in_layout(False)
                tip.set_visible(False)
                self._tips[i] = tip
            return self._tips[i]

        def _tip_placement(self, tip, ax, x, y, pad=8):
            """Which side of the point the box goes on.

            Measured, not guessed: the box is compared against the room left in
            the axes, and put on whichever side it fits. So a read-out near the
            right-hand end of a curve — where the interesting frames usually
            are — appears to the *left* of the cursor instead of hanging off
            the panel. Returns (dx, dy, ha, va) for an offset-points
            annotation."""
            box = ax.get_window_extent()
            px, py = ax.transData.transform((x, y))
            try:
                size = tip.get_window_extent(self.get_renderer())
                w, h = size.width, size.height
            except Exception:                # no renderer yet: assume roomy
                w = h = 0
            flip_x = px + pad + w > box.x1 and px - pad - w >= box.x0
            flip_y = py + pad + h > box.y1 and py - pad - h >= box.y0
            return (-pad if flip_x else pad, -pad if flip_y else pad,
                    "right" if flip_x else "left",
                    "top" if flip_y else "bottom")

        def _hide_tips(self):
            changed = False
            for tip in self._tips:
                if tip is not None and tip.get_visible():
                    tip.set_visible(False)
                    changed = True
            if changed:
                self.draw_idle()

        def _hover(self, event):
            """Show the real (un-normalized) value under the cursor.

            The curve may be drawn as a ratio, so the number on the axis is not
            the number the user measured. Hovering brings the measured one back
            without needing a second plot."""
            ax = event.inaxes
            if ax is None or event.xdata is None or self.worm is None:
                self._hide_tips()
                return
            try:
                i = list(self.axes).index(ax)
            except ValueError:
                self._hide_tips()
                return
            drawn = self._drawn[i] if i < len(self._drawn) else {}
            if not drawn:
                self._hide_tips()
                return
            f = min(drawn, key=lambda k: abs(k - event.xdata))
            if abs(f - event.xdata) > max(1.0, 0.02 * (max(drawn) - min(drawn) or 1)):
                self._hide_tips()
                return
            # only when the cursor is actually near the curve, not anywhere
            # in the panel — otherwise the box follows the mouse everywhere
            y = drawn[f]
            lo, hi = ax.get_ylim()
            if event.ydata is None or abs(event.ydata - y) > 0.18 * abs(hi - lo):
                self._hide_tips()
                return
            real = self._raw[i].get(f)
            unit = self._units[i] or self.metrics[i]
            text = f"frame {f}\n{unit}: {self._fmt(real)}" if real is not None \
                else f"frame {f}"
            tip = self._tip(i)
            tip.xy = (f, y)
            tip.set_text(text)
            # Put the box on whichever side of the cursor has room for it.
            dx, dy, ha, va = self._tip_placement(tip, ax, f, y)
            tip.set_horizontalalignment(ha)
            tip.set_verticalalignment(va)
            tip.xyann = (dx, dy)
            for other in self._tips:         # one box at a time
                if other is not None and other is not tip:
                    other.set_visible(False)
            tip.set_visible(True)
            self.draw_idle()
else:
    class PlotPanel(QLabel):
        frameSelected = pyqtSignal(int)

        def __init__(self, metrics):
            super().__init__("Install matplotlib to see plots")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        def set_worm(self, worm):
            pass

        def set_frame(self, f):
            pass

        def set_smooth(self, on):
            pass

        def set_scales(self, scales):
            pass

        def set_normalized(self, names):
            pass
