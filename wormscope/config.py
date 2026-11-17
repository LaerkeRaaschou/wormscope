"""
All tunable constants in one place.
"""

import re

# --------------------------------------------------------------------------- #
# Colour theme: light grayish-blue surfaces, dark-blue accents
# --------------------------------------------------------------------------- #
BG_LIGHT = "#e4eaf2"          # window background
PANEL = "#eff3f9"             # group-box / plot background
DARK_BLUE = "#1f3a63"         # accents: buttons, borders, text, list selection
DARK_BLUE_HI = "#2b4f86"      # hover / accent-light
DARK_BLUE_LO = "#16294a"      # pressed
GRIDCOL = "#c4d0e2"           # borders, grid, slider groove
PLOT_LINE = "#4372a3"         # metric curve
PLOT_DOT = "#14243f"          # current-frame dot
PLOT_AX_BG = "#f7fafe"        # plot axes background

# Worm mask overlay colours (RGB) and blend strength.
MASK_UNSEL_RGB = (120, 165, 215)     # unselected worms
MASK_SEL_RGB = (26, 58, 120)         # selected worm
MASK_ALPHA = 0.5                     # overlay opacity
SKEL_RGB = (230, 60, 60)             # selected worm's skeleton

PANEL_BG_RGB = (214, 224, 236)       # letterbox behind the image
VIEWPORT_RGB = (255, 210, 90)        # zoom-region rectangle drawn on the video
SCALE_LINE_RGB = (255, 210, 90)      # the line drawn on a scale bar (Units…)

# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
PLAYBACK_FPS = 10          # display speed for playback (not the acquisition FPS)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PHOTO_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}  # a single image to upload
MAX_PLOTS = 4
SPEED_METRIC = "speed"                 # centroid speed, px/frame
ACCEL_METRIC = "acceleration"          # change of speed, px/frame^2
KINEMATIC_METRICS = [SPEED_METRIC, ACCEL_METRIC]
SMOOTH_WINDOW = 5                      # moving-average window for Smooth toggle
ZOOM_MIN, ZOOM_MAX = 0.1, 40.0
ZOOM_STEP = 1.25                     # one − / + zoom step (buttons and keys)
PAN_STEP = 0.15                      # one pan-key step, as a fraction of the view
UNDO_LIMIT = 100                       # how many actions the undo stack keeps
HIT_RADIUS = 45                        # px: centroid fallback when clicking a worm
BRUSH_MIN, BRUSH_MAX, BRUSH_DEFAULT = 2, 40, 8
# Default tool when new worm
NEW_WORM_TOOL = "sam"


# --------------------------------------------------------------------------- #
# SAM2 (optional)
# --------------------------------------------------------------------------- #
# Hand-picked files, used if the GUI doesn't remember any. Normally left empty:
# the model below is then downloaded and used automatically.
SAM2_CFG = ""      # e.g. "/Users/you/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = ""     # e.g. "/Users/you/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_DEVICE = ""   # "" = auto-detect (cuda -> mps -> cpu)
SAM2_MODEL = "small"  # tiny / small / base_plus / large — downloaded on first use

# How many frames' image embeddings to keep
SAM2_EMBED_CACHE = 16

# --------------------------------------------------------------------------- #
# Filename patterns
# --------------------------------------------------------------------------- #
LAST_INT_RE = re.compile(r"(\d+)(?!.*\d)")          # last run of digits in a name
WORM_RE = re.compile(r"worm[_-]?(\d+)", re.IGNORECASE)
