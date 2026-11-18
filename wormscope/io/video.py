"""Turning a video file (.mp4, .avi, .mov, …) into a folder of frames.

wormscope works on frames, so an uploaded video is decoded once into
``<name>_wormscope/frames/frame_00000.png, frame_00001.png, …`` (lossless
PNG, the frame number is the file's trailing integer). From then on it is an
ordinary video folder: segment it, measure it, Save, and reopen it with
“Continue editing video…” — the video file itself is never needed again and
never modified.

``source_video.json`` next to frames/ records where the frames came from (the
file, its size and date, the frame count and frame rate). Uploading the same,
unchanged video again reuses those frames instead of decoding it again, as
long as nothing was saved in that folder yet.

Decoding uses imageio's ffmpeg plugin (imageio-ffmpeg ships its own ffmpeg).
Nothing here imports Qt: progress is reported through a plain callback.
"""

import json
import os
import shutil
from pathlib import Path

from PIL import Image

from ..config import VIDEO_EXTS

SOURCE_FILE = "source_video.json"
FRAME_NAME = "frame_{:05d}.png"


class VideoCancelled(Exception):
    """The user stopped the decoding; nothing was kept."""


def is_video(path) -> bool:
    """A video file wormscope can decode into frames (.mp4 .avi .mov …)."""
    path = Path(path)
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def _source_record(video: Path) -> dict:
    st = video.stat()
    return {"video": str(video.resolve()), "name": video.name,
            "bytes": st.st_size, "mtime": int(st.st_mtime)}


def read_source(folder) -> dict:
    """source_video.json of a video folder, or {}."""
    p = Path(folder) / SOURCE_FILE
    try:
        data = json.loads(p.read_text()) if p.is_file() else {}
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def frames_from(folder, video) -> bool:
    """True when `folder`/frames already holds every frame of this very
    video (same file, size and date), so decoding it again can be skipped."""
    rec = read_source(folder)
    if not rec:
        return False
    cur = _source_record(Path(video))
    if any(rec.get(k) != cur[k] for k in ("name", "bytes", "mtime")):
        return False
    n = rec.get("frames") or 0
    frames = Path(folder) / "frames"
    return n > 0 and all((frames / FRAME_NAME.format(i)).is_file()
                         for i in (0, n - 1))


def _open(video):
    import imageio.v2 as imageio            # raised to the caller if missing
    return imageio.get_reader(str(video), format="FFMPEG")


def video_meta(video) -> tuple[float | None, int]:
    """(frames per second or None, estimated frame count or 0 if unknown)."""
    reader = _open(video)
    try:
        meta = reader.get_meta_data()
    finally:
        reader.close()
    fps = meta.get("fps") or None
    n = meta.get("nframes")
    if not isinstance(n, (int, float)) or n != n or n in (float("inf"), 0):
        duration = meta.get("duration") or 0
        n = round(duration * fps) if fps and duration else 0
    return (float(fps) if fps else None), int(n)


def extract_frames(video, out_dir, progress=None):
    """Decode `video` into `out_dir`/frames/ as numbered PNGs.

    `progress(done, total)` is called as frames are written (total is an
    estimate, 0 when unknown) and may return True to stop: the half-written
    frames are then removed and VideoCancelled raised. Frames are written to a
    temporary folder and only renamed to frames/ once complete, so a stopped or
    failed decode never leaves a folder that looks finished.
    Returns (number of frames, fps or None)."""
    video, out_dir = Path(video), Path(out_dir)
    fps, total = video_meta(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "frames.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    n = 0
    reader = _open(video)
    try:
        for frame in reader:
            Image.fromarray(frame).save(tmp / FRAME_NAME.format(n),
                                        compress_level=1)   # lossless, fast
            n += 1
            if progress is not None and progress(n, max(total, n)):
                raise VideoCancelled(f"stopped after {n} frames")
    except BaseException:
        reader.close()
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    reader.close()
    if n == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"no frames could be read from {video.name}")
    frames = out_dir / "frames"
    if frames.exists():                      # stale frames of an unsaved session
        shutil.rmtree(frames)
    os.replace(tmp, frames)
    (out_dir / SOURCE_FILE).write_text(json.dumps(
        {**_source_record(video), "frames": n, "fps": fps}, indent=2))
    print(f"[video] {video.name}: {n} frames"
          + (f" at {fps:g} fps" if fps else "") + f" -> {frames}")
    return n, fps
