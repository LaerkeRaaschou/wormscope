#!/usr/bin/env python3
"""wormscope entry point.

    python main.py                       # empty window: open a folder from the GUI
    python main.py path/to/video_folder  # a folder with frames/ (and binary/)
    python main.py path/to/raw_frames    # a plain folder of images

Either the folder you name or nothing at all — no demo data, no default folder.

Everything else lives in the wormscope/ package — see wormscope/README.md for
the module map.

Deps:  pip install PyQt6 matplotlib pillow numpy scikit-image
Optional:  imageio + imageio-ffmpeg (video export), sam2 + torch (SAM2 tools)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from anywhere: make sure this folder is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wormscope.app import run  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run(path))


if __name__ == "__main__":
    main()
