"""
Launch of wormscope GUI:
    python main.py                       # empty window: open a folder from the GUI
    python main.py path/to/video_folder  # a folder with frames/ (and binary/)
    python main.py path/to/raw_frames    # a plain folder of images
"""

#from __future__ import annotations

import sys
from pathlib import Path

# Run from anywhere: make sure this folder is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wormscope.app import run

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None # Get path from command
    run(path) # Start the app


if __name__ == "__main__":
    main()
