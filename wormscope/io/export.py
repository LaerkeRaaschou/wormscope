"""Rendering the corrected overlay out to an .mp4 (or PNG frames)."""

from pathlib import Path

import numpy as np

from ..config import PLAYBACK_FPS, SKEL_RGB
from ..core.utils import color_for
from ..render import blend


def export_frame(frames, worms, masks, f):
    """Original frame + per-worm coloured masks + red skeletons (reflects the
    current, edited masks)."""
    out = frames.get(f).copy()
    for w in worms:
        bbox = w.bbox(f)
        if bbox is None:
            continue
        blend(out, masks.get(w.obj_id, f), bbox,
              color_for(w.obj_id).astype(np.float32), 0.5)
    for w in worms:
        sk = masks.skel(w.obj_id, f)
        if sk is not None and sk.any():
            out[sk] = np.array(SKEL_RGB, np.uint8)
    return out


def export_overlay(frames, worms, masks, n_frames, out_dir, idx_to_key=None,
                   fps=None, name="corrected_overlay", progress=None):
    """Write every frame's overlay to <out_dir>/<name>.mp4.

    Falls back to a folder of PNGs when the imageio ffmpeg backend is missing
    (imageio would otherwise mis-route the .mp4 to the tifffile plugin and crash
    on the 'fps' argument).

    `progress(done, total)` is called after each frame and may return True to
    stop early. Returns (target_path, wrote_video: bool, frames_written: int).
    Raises ImportError if imageio itself is not installed."""
    import imageio                              # raised to the caller as-is

    out_dir = Path(out_dir)
    idx_to_key = idx_to_key or {}
    fps = fps or PLAYBACK_FPS

    writer = None
    try:
        writer = imageio.get_writer(str(out_dir / f"{name}.mp4"), format="FFMPEG",
                                    fps=fps, codec="libx264", macro_block_size=None)
    except Exception as err:
        print(f"[export] ffmpeg writer unavailable ({err}); writing PNGs")

    written = 0
    try:
        if writer is not None:
            for f in range(n_frames):
                writer.append_data(export_frame(frames, worms, masks, f))
                written += 1
                if progress is not None and progress(written, n_frames):
                    break
            writer.close()
            return out_dir / f"{name}.mp4", True, written
        frames_dir = out_dir / name
        frames_dir.mkdir(parents=True, exist_ok=True)
        for f in range(n_frames):
            key = idx_to_key.get(f, f)
            imageio.imwrite(str(frames_dir / f"{int(key):05d}.png"),
                            export_frame(frames, worms, masks, f))
            written += 1
            if progress is not None and progress(written, n_frames):
                break
        return frames_dir, False, written
    except Exception:
        if writer is not None:
            writer.close()
        raise
