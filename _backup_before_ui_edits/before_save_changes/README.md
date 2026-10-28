# wormscope

GUI for reviewing and correcting worm segmentation from the Worm_vision pipeline:
step through frames, inspect per-worm metrics, fix masks with a brush or SAM2
point prompts, and write the corrections back to `binary/`, `skel/` and
`measurements.csv`.

```bash
python main.py                       # empty window — open a folder from the GUI
python main.py path/to/video_folder  # a folder containing frames/ (and binary/)
python main.py path/to/raw_frames    # a plain folder of images -> segment with SAM2
```

**Either the folder you name, or nothing at all.** A plain start opens an empty
window with only **Upload new video frames…** and **Continue editing video…** live — no demo data,
no default folder.

Deps: `pip install PyQt6 matplotlib pillow numpy scikit-image`
Optional: `imageio` + `imageio-ffmpeg` (overlay video export), `sam2` + `torch`
(SAM2 tools — see below).

## Segmenting with SAM2

Typical pass over a clip:

1. Select a worm, or press **Add** — a new worm starts with **+** (AI features) already
   selected (`NEW_WORM_TOOL` in `config.py`; it falls back to the brush when the
   frames aren't on disk, since SAM2 needs them). Click on the worm.
   **Shift+click** places the opposite sign for that one click, so a stray
   negative point costs no trip to the toolbar. **−** switches the default
   the other way round permanently — every plain click is then negative, and
   Shift gives you a positive one.
2. **Segment ▶ all** tracks every prompted worm through the whole clip, one
   worm at a time.
3. Find a frame it got wrong and correct it, either way:
   * click more SAM points on that frame (Shift for negative), or
   * fix the mask with the **brush** — no extra step needed, see *anchoring*
     below — or press **Mask → prompt** to hand it over straight away.
   **Draw** and **Erase** sit one above the other, and **Shift+drag borrows the
   other one** for a single stroke: paint the body, hold Shift to trim the edge
   you overshot, let go, keep painting. The toolbar never changes underneath
   you, and one stroke is always one tool — releasing Shift mid-drag does not
   flip it halfway.
4. **Re-segment ▶ here** re-tracks *that worm* from the corrected frame onwards;
   tick **back** to also re-track towards frame 0 (for when the worm was already
   wrong before the frame you fixed).

### Why corrections on several frames don't fight each other

**One SAM2 session per worm.** Each worm gets its own `inference_state`, created
on first use, so a prompt or a re-track on one worm can never disturb another's
memory or masks — correcting worm 3 at frame 47 leaves worms 1, 2 and 4 exactly
as they were. The decoded frames (the big allocation) are shared between
sessions, and so is the image-embedding cache, since an embedding depends only
on the frame and the model. Deleting a worm drops its session with it.

**Hand edits become anchors.** A frame you painted is invisible to the tracker
until it is a prompt, so a re-track would simply overwrite it. wormscope
remembers which frames you edited and feeds each one to SAM2 as a mask prompt
(`add_new_mask`) *before* tracking starts. The correction survives, and the
tracker now works outwards from it — which is what makes prompts on several
frames reinforce each other instead of competing. Anchored frames aren't
re-anchored on later runs; **Clear pts** turns them back into pending edits.

### It runs in the background

SAM2 lives on a worker thread, so **the GUI never freezes while it thinks**:
paint, step frames, undo, switch worms — all of it keeps working while a
tracking run is in flight, and masks appear frame by frame as they arrive.
Further SAM2 requests simply queue behind the running one.

If you edit a frame *while* a run is passing over it, your mask wins: the
tracker's result for that frame is dropped and the info panel says how many
frames were kept that way.

### “lock edits”: corrections that stay corrected

Tick **lock edits** (Zoom panel, right under the brush) and *every* frame you have
fixed by hand becomes read-only to SAM2 — not just the ones you happened to
edit mid-run. Those frames are still sent to SAM2 as `add_new_mask` prompts, so
they keep steering the track exactly as before; the tracker simply never writes
its own answer back over them.

Use it once the frames you care about are right and you want to re-run tracking
to fix the rest without watching your work disappear. Untick it and SAM2 owns
every frame again.

Progress shows in a small panel at the **bottom of the Zoom panel's control
column**, right under the buttons that start the work, with a **Stop** button —
tracking, exporting and pre-encoding all use it. Stopping keeps the frames
already done rather than throwing them away.

It reports a **percentage**, not `47 / 98`: the frame total is an estimate (a
run can stop early, and a worm's prompts need not span the whole clip), so a
count can march past its own total. A percentage of an estimate at least admits
to being one.

**The model loads as soon as you open a video.** Building the predictor and
embedding frame 0 is most of the cost of the first click, so wormscope does it
in the background while you are still looking at the frame.

**Embeddings are cached.** SAM2 itself keeps only the *most recent* frame's
image embedding, so without help every frame you return to is re-encoded — the
slow part of a click. wormscope keeps the last `SAM2_EMBED_CACHE` frames (16 by
default, see `config.py`); raise it if you have the memory and jump around a
lot. Encode times, cache hits and misses are printed to the terminal:

```
[sam2] embed frame 12: 0.38s (miss, cache 4/16)
[sam2] add_points frame 12 obj 1 (2 pts): 0.06s
[sam2] propagated 118 frames (forwards from 12) in 24.3s
```

## SAM2 setup (once, not every launch)

**You are asked once, when SAM2 is first needed.** A file picker opens for the
config (`*.yaml`) and the checkpoint (`*.pt`); both are written to
`~/.wormscope/settings.json` and reused on every future launch — no
`export SAM2_CFG=…` before starting the program, and no button cluttering the
toolbar for a thing you do once. If the pair turns out not to load, the error
dialog offers the picker again.

To change them later without the GUI, see `python -m wormscope.settings` below.

Where a path comes from, highest priority first:

1. the settings file (`~/.wormscope/settings.json`, what the button writes);
2. the `SAM2_CFG` / `SAM2_CKPT` environment variables;
3. `SAM2_CFG` / `SAM2_CKPT` in `wormscope/config.py` (bake them into the project).

**Why the config is special.** SAM2 composes its config with
[Hydra](https://hydra.cc) (`hydra-core`, installed alongside `sam2`), and
`sam2/__init__.py` initialises it with the *sam2 package itself* as the config
search path. Hydra therefore expects a package-relative **name** such as
`configs/sam2.1/sam2.1_hiera_l.yaml` — handed an absolute path it strips the
leading slash and fails with `Cannot find primary config 'Users/…'`. wormscope
converts the file you pick into that name automatically
(`segmentation.hydra_config_name`): a config inside the sam2 package becomes its
package-relative name, and one stored anywhere else keeps its bare filename while
its folder is added to Hydra's search path. So you can pick the `.yaml` in a file
dialog like any other file. The configs ship inside the package, e.g.
`.venv/lib/python3.13/site-packages/sam2/configs/sam2.1/`.

The config and checkpoint must be a matching pair (`sam2.1_hiera_l.yaml` goes with
`sam2.1_hiera_large.pt`); wormscope warns up front when they clearly aren't,
instead of letting it fail later with a wall of missing state-dict keys.

Either form works everywhere a config is accepted — an absolute path (what the
file dialog gives you) or a Hydra name like `configs/sam2.1/sam2.1_hiera_l.yaml`.

To inspect or change the saved values without opening the GUI:

```bash
python -m wormscope.settings                       # what is set, and what wins
python -m wormscope.settings --set-sam2 configs/sam2.1/sam2.1_hiera_l.yaml \
                                       ~/Desktop/job/sam2.1_hiera_large.pt
python -m wormscope.settings --clear-sam2          # fall back to env vars again
```

Because the settings file outranks the environment, `export SAM2_CFG=…` has no
effect while a different value is saved — the app prints a NOTE saying so rather
than silently ignoring it.

A saved path that no longer exists is skipped, so moving the checkpoint just
falls back to the next source instead of failing. `sam2_device` in the settings
file (or `SAM2_DEVICE`) forces `cuda` / `mps` / `cpu`; empty means auto-detect.
Point `WORMSCOPE_SETTINGS` at another file to keep per-project settings.

## Calibrating units

**Units…** (Plots panel) opens the calibration dialog. Load an image of a
scale (a stage micrometer, or any image with a scale bar, including 16-bit
TIFFs), or use the current frame. Drag a line along a stretch of known length
and enter that length in nm, µm or mm. µm per pixel = known length ÷ line
length, and you can also type it directly. Drag an end to adjust it, and hold
Shift to keep the line horizontal or vertical. Set frames per second in the
same dialog. **Use pixels** clears the calibration.

Measurements are always stored in pixels. The factor is applied only when
showing the plots and Info, and when writing the µm columns on Save, so
recalibrating never changes the data. The scale image must have the same pixel
size as the video. The dialog warns when its dimensions differ. Reopening the
dialog keeps the image, the line and the known length.

## Reading the plots

Four stacked curves for the selected worm: **area**, **length**, **speed** and
**acceleration**. Click or drag anywhere in them to scrub through frames; the
dot marks where you are.

Area and length are drawn **divided by their own median**, so the y-axis reads
around 1 and the curve answers the question you actually have — *is the worm
bigger or smaller than usual right now?* A frame at 1.9 is a worm that doubled,
whatever the calibration says a pixel is worth, and a bad segmentation shows up
as a spike instead of hiding inside a large absolute number. Speed and
acceleration keep their own units, where zero and the sign are the meaning.

The real number is never lost: **hover a curve** and a small box shows the frame
and the true value, in whatever units **Units…** is set to. The box is placed on
whichever side of the cursor has room for it — measured against the axes, so at
the right-hand end of a curve it appears to the left instead of hanging off the
panel — and it is kept out of the layout calculation, so the plot never shifts
or shrinks to accommodate it. A worm whose median is zero or missing is drawn
unchanged rather than divided.

Normalization and smoothing are display only — `measurements.csv` always holds
the raw measured values.

## Expected data layout

```
data/<video>/
    frames/            ORIGINAL frames        -> shown underneath the masks
    binary/            worm{id:02d}_{stem}.png -> per-worm masks (EDITABLE)
    skel/              worm{id:02d}_{stem}.png -> skeletons (recomputed on save)
    measurements.csv   per-frame metrics       -> Plots panel (rewritten on Save)
    measurements_plot.png
```

Frame alignment keys on the trailing integer in each filename, so frames, masks
and CSV rows line up even when their name prefixes differ. If a folder you point
at has no frames, the window opens empty rather than substituting anything.

## Module map

```
main.py                    entry point: parse the path argument, call wormscope.app.run
wormscope/
├── __init__.py            package docstring; applies the Qt plugin-path fix first
├── qt_setup.py            macOS/conda fix: point Qt at PyQt6's bundled plugins
├── config.py              ALL constants: colours, metric names, limits, regexes
├── settings.py            remembered user paths (~/.wormscope/settings.json)
├── theme.py               palette + resources/style.qss loader (no Qt import)
├── resources/style.qss    the stylesheet itself, with $palette placeholders
├── app.py                 choose a dataset, start QApplication, show the window
├── render.py              pure-numpy compositing: mask blend, outlines
├── core/                  data + maths, no Qt — safe to use from scripts/tests
│   ├── models.py          Worm, FrameSource, MaskStore, Dataset
│   ├── geometry.py        skeletonize, mask_stats, brush stamping
│   ├── kinematics.py      centroid speed / acceleration
│   ├── metrics.py         per-worm metric refresh + unit scaling
│   └── utils.py           moving average, filename parsing, CSV column matching
├── io/                    getting data in and out
│   ├── loaders.py         load_dataset / build_raw_dataset and their helpers
│   ├── demo.py            make_demo — synthetic dataset used by the tests
│   ├── writers.py         measurements.csv + measurements_plot.png
│   └── export.py          corrected-overlay .mp4 (PNG fallback)
├── segmentation/          swappable segmentation backends
│   ├── __init__.py        SEGMENTER_FACTORY / make_segmenter
│   ├── sam2_backend.py    Sam2Backend (torch + sam2 imported lazily)
│   └── feature_cache.py   LRU over per-frame image embeddings + timings
├── widgets/               reusable Qt widgets, no app logic
│   ├── video_view.py      VideoView   — whole frame + viewport rectangle
│   ├── zoom_view.py       ZoomView    — zoom/pan/paint canvas (emits signals only)
│   ├── plot_panel.py      PlotPanel   — stacked metric plots, scrubbable,
│   │                                    area/length ÷ median with a hover read-out
│   ├── image_canvas.py    ImageCanvas — zoom/pan/fit base for tool canvases
│   ├── scale_view.py      ScaleView   — draw a line on a scale bar (calibration)
│   ├── imaging.py         numpy -> QImage
│   └── colors.py          QColors for hand-painted bits (letterbox, viewport box)
└── ui/                    the main window, one mixin per area
    ├── main_window.py     state, frame navigation, selection, playback, keys
    ├── layout.py          builds EVERY widget and lays out the four panels
    ├── rendering.py       overlay compositing, skeleton, image adjustments
    ├── editing.py         brush tools, stroke handling, metric refresh
    ├── history.py         undo / redo stack
    ├── progress.py         inline progress panel (never modal)
    ├── sam_worker.py       SAM2 on a worker thread: job queue + signals
    ├── worms.py           add / delete / reassign worm IDs
    ├── sam.py             SAM2 point prompts and propagation
    ├── io_actions.py      open, import, units, save, export
    └── calibration_dialog.py  Units…: scale image + line -> µm/pixel, and fps
tests/
├── test_parity.py             drives old + new through the same actions, compares pixels/CSV
├── test_dataset_roundtrip.py  loads a fake data folder in both, compares files written
├── test_settings.py           SAM2 path resolution order, picker, persistence
├── test_sam2_config.py        absolute path -> Hydra config name conversion
├── test_skeleton_display.py   no skeleton on a frame that has no mask
├── test_sam_workflow.py       +/- switch, mask prompts, re-tracking, lock edits, cache
├── test_startup.py            empty start, and that a named folder still loads
├── test_plots.py              median-normalized curves, hover read-out, % progress
└── test_editing_tools.py      Shift swaps draw/erase, arrow keys, N, removed buttons
wormscope_gui3.py          the original single-file version, kept for reference
```

## Where to make a change

| I want to… | Edit |
|---|---|
| change a colour, limit or metric name | `config.py` |
| make the controls smaller / images bigger | `CONTROL_H` and the row stretches in `ui/layout.py` |
| restyle buttons, sliders, group boxes | `resources/style.qss` (colours come from `theme.PALETTE`) |
| change a hand-painted colour (letterbox, viewport box) | `PANEL_BG_RGB` / `VIEWPORT_RGB` in `config.py` |
| add a button, slider or panel | `ui/layout.py` (widget) + the matching mixin (behaviour) |
| change what the overlay shows | `ui/rendering.py`, pixel helpers in `render.py` |
| change brush / editing behaviour | `ui/editing.py`, `core/geometry.py` |
| change what a modifier key does on the canvas | `widgets/zoom_view.py` (`paint_mode`, `_sam_label`) |
| add a keyboard shortcut | `LayoutMixin._build_shortcuts` (works whatever has focus) or `MainWindow.keyPressEvent` |
| add an undoable action | push `(undo_fn, redo_fn)` via `_push_action` — see `ui/history.py` |
| support a new folder layout or CSV | `io/loaders.py` |
| change what Save writes | `io/writers.py`, `ui/io_actions.py` |
| change which tool a new worm starts with | `NEW_WORM_TOOL` in `config.py` |
| change SAM2 prompting or tracking | `ui/sam.py` (GUI) + `segmentation/sam2_backend.py` (API calls) |
| swap in another segmentation model | implement `add_points` / `propagate` / `reset`, set `wormscope.segmentation.SEGMENTER_FACTORY` |
| change how µm/pixel is calibrated (scale image + line) | `ui/calibration_dialog.py`, maths in `core/metrics.py` (`um_per_px`, `UNIT_TO_UM`) |
| change how pixels become µm in plots / CSV | `core/metrics.py` (`unit_scales`), `io/writers.py` |
| change how area / length are measured | `core/geometry.py` (`mask_stats`), `core/metrics.py` |
| remember another path or preference | `settings.py` (JSON on disk) + a picker in `ui/` |
| change startup / dataset selection | `app.resolve_dataset` (or `main.py` for the CLI) |

Rules of thumb that keep this easy to extend:

- A frame with no mask has no skeleton. `skel/` often still holds PNGs from an
  earlier pipeline run for worm-frames whose mask was later deleted or never
  segmented; `MaskStore.skel()` returns None unless the mask is really there, so
  the red skeleton always corresponds to something you can see and edit.
- `core/` and `io/` never import Qt, so metrics and loaders stay scriptable and testable.
- Widgets in `widgets/` report what the user did; only `ui/` changes data.
- The main window owns state; mixins are behaviour grouped by topic, so adding a
  feature usually means one new method in one file.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pre-commit install --install-hooks
```

`requirements.txt` covers the GUI, the optional video export and the dev tools.
SAM2 and torch are not in it — see *SAM2 setup* above.

### The hooks

`pre-commit install --install-hooks` wires up two stages, both configured in
`.pre-commit-config.yaml`:

| When | What runs | Roughly |
|---|---|---|
| every commit | whitespace/EOF/merge-marker hygiene, `ruff check`, then `tests/unit` + `tests/test_contracts.py` | seconds |
| push, on `main` only | the whole suite, coverage floor included (`scripts/pytest_push.sh`) | minutes |

Pushing any other branch skips the suite: side branches are for work in
progress. The check reads the branch you are standing on, so
`git push origin side:main` slips past it.

Nothing reformats your code. The hygiene hooks touch whitespace only, and ruff
runs without `--fix` — it reports, you decide. Its rules live in `ruff.toml` and
are deliberately narrow (`E`, `F`, `W`): dead imports, undefined names, stray
whitespace. `wormscope_gui3.py` is excluded everywhere, so the frozen original
stays byte-for-byte as the parity tests expect.

Useful:

```bash
pre-commit run --all-files    # run every commit-stage hook over the whole repo
ruff check .                  # just the linter
git commit -n                 # skip the hooks for one commit, knowingly
```

## Tests

Both scripts compare this package against `wormscope_gui3.py`, so they only run
where the original file is still present:

```bash
QT_QPA_PLATFORM=offscreen python tests/test_parity.py wormscope_gui3.py
QT_QPA_PLATFORM=offscreen python tests/test_dataset_roundtrip.py wormscope_gui3.py
QT_QPA_PLATFORM=offscreen python tests/test_settings.py     # standalone
python tests/test_sam2_config.py                            # standalone
QT_QPA_PLATFORM=offscreen python tests/test_skeleton_display.py   # standalone
QT_QPA_PLATFORM=offscreen python tests/test_sam_workflow.py      # standalone
QT_QPA_PLATFORM=offscreen python tests/test_startup.py           # standalone
QT_QPA_PLATFORM=offscreen python tests/test_plots.py             # standalone
QT_QPA_PLATFORM=offscreen python tests/test_editing_tools.py     # standalone
```

`test_settings.py` uses a throwaway settings file, so running it never touches
your real `~/.wormscope/settings.json`.

`test_parity.py` drives both versions through the same script of user actions —
stroke, erase, undo/redo, propagate, add/reassign/delete worm, unit calibration,
image adjustments — and asserts the rendered pixels, metrics and CSV are byte-for-byte
identical. `test_dataset_roundtrip.py` builds a small fake pipeline folder, edits and
saves in both, and compares every file written to disk.

## Keyboard

| Key | Action |
|---|---|
| ← / → | previous / next frame — from anywhere in the window |
| Shift+← / → | jump ten frames |
| Space | play / pause |
| N | new worm (armed with the + tool, ready to click) |
| Shift+click | flip the SAM point's sign for that one click |
| Shift+drag | Draw erases / Erase draws, for that one stroke |
| + / − | zoom the editing canvas in / out (also in the Units… dialog) |
| ⌘Z or Ctrl+Z | undo |
| ⌘⇧Z or Ctrl+Shift+Z (also Ctrl+Y) | redo |
| W A S D, or Alt/Option + arrows | pan the editing canvas |
| Ctrl+S | save |

← / → are window-wide shortcuts rather than plain key presses, and buttons and
sliders hand focus back after a click. Otherwise the last control you touched
would quietly eat the keyboard — dragging the brush slider once and then pressing
← would nudge the brush instead of stepping back a frame.
