"""All tunable constants in one place.

Nothing here imports Qt or numpy, so any module (including tests) can read the
configuration without pulling in the GUI stack.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Colour theme: light grayish-blue surfaces, dark-blue accents
# --------------------------------------------------------------------------- #
BG_LIGHT = "#e4eaf2"          # window background (grayish blue)
PANEL = "#eff3f9"             # group-box / plot background (lighter)
DARK_BLUE = "#1f3a63"         # accents: buttons, borders, text, list selection
DARK_BLUE_HI = "#2b4f86"      # hover / accent-light
DARK_BLUE_LO = "#16294a"      # pressed
GRIDCOL = "#c4d0e2"           # borders, grid, slider groove
PLOT_LINE = "#4372a3"         # metric curve (steel blue)
PLOT_DOT = "#14243f"          # current-frame dot (deep navy)
PLOT_AX_BG = "#f7fafe"        # plot axes background

# Worm mask overlay colours (RGB) and blend strength.
MASK_UNSEL_RGB = (120, 165, 215)     # light mid-blue  (unselected worms)
MASK_SEL_RGB = (26, 58, 120)         # darker blue     (selected worm)
MASK_ALPHA = 0.5                     # overlay opacity
SKEL_RGB = (230, 60, 60)             # selected worm's skeleton (red)
ONION_RGB = (240, 150, 60)           # onion-skin: previous frame's mask (orange)
ONION_ALPHA = 0.35                   # onion-skin opacity

PANEL_BG_RGB = (214, 224, 236)       # letterbox behind the image (grayish blue)
VIEWPORT_RGB = (255, 210, 90)        # zoom-region rectangle drawn on the video

# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
PLAYBACK_FPS = 10          # display speed for playback (not the acquisition FPS)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_PLOTS = 4
SPEED_METRIC = "speed"                 # centroid speed, px/frame
ACCEL_METRIC = "acceleration"          # change of speed, px/frame^2
KINEMATIC_METRICS = [SPEED_METRIC, ACCEL_METRIC]
SMOOTH_WINDOW = 5                      # moving-average window for Smooth toggle
ZOOM_MIN, ZOOM_MAX = 0.1, 40.0
UNDO_LIMIT = 100                       # how many actions the undo stack keeps
HIT_RADIUS = 45                        # px: centroid fallback when clicking a worm
BRUSH_MIN, BRUSH_MAX, BRUSH_DEFAULT = 2, 40, 8
# Which tool "Add" arms for a brand-new worm: "sam" (click the animal) or
# "draw" (paint it by hand). "sam" falls back to "draw" when the frames aren't
# on disk, since SAM2 can't run without them.
NEW_WORM_TOOL = "sam"


# --------------------------------------------------------------------------- #
# SAM2 (optional)
# --------------------------------------------------------------------------- #
# Last-resort defaults for the SAM2 config + checkpoint. Normally you don't
# touch these: pick the files once with the GUI's “SAM2 files…” button and they
# are remembered in ~/.wormscope/settings.json (see wormscope/settings.py).
# Fill them in here only if you want the paths baked into the project itself.
SAM2_CFG = ""      # e.g. "/Users/you/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = ""     # e.g. "/Users/you/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_DEVICE = ""   # "" = auto-detect (cuda -> mps -> cpu)

# How many frames' image embeddings to keep. SAM2 itself keeps exactly one, so
# every frame change re-runs the image encoder; this is the LRU that avoids it.
# Bigger = fewer re-encodes but more memory (hundreds of MB per handful of
# frames at hiera-L). Raise it if you have the RAM and jump around a lot.
SAM2_EMBED_CACHE = 16

# --------------------------------------------------------------------------- #
# Filename patterns
# --------------------------------------------------------------------------- #
LAST_INT_RE = re.compile(r"(\d+)(?!.*\d)")          # last run of digits in a name
WORM_RE = re.compile(r"worm[_-]?(\d+)", re.IGNORECASE)
