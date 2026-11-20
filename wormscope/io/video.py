"""Turning a video file (.mp4, .avi, .mov, …) into a folder of frames.

wormscope works on frames, so an uploaded video is decoded once into
``<name>_wormscope/frames/frame_00000.png, frame_00001.png, …`` (lossless
PNG). The number in a file name is always the frame's number *in the video*
(0 = the first frame), also when only some frames are kept, so the frame
column of every CSV points straight back to the video.

Which frames are kept is the `sampling` (chosen in the upload dialog,
ui/video_import_dialog.py)::

    {"mode": "all"}                               every frame, original fps
    {"mode": "every", "step": 4}                  every 4th frame: fps / 4
    {"mode": "random", "count": 200, "seed": 1}   200 frames at random, in
                                                  time order (no frame rate)

From then on it is an
ordinary video folder: segment it, measure it, Save, and reopen it with
“Continue editing video…” — the video file itself is never needed again and
never modified.

``source_video.json`` next to frames/ records where the frames came from (the
file, its size and date, the sampling, the frame count, the video's frame
rate and the frame rate of the kept frames). Uploading the same, unchanged
video again with the same sampling reuses those frames instead of decoding it
again, as long as nothing was saved in that folder yet.

Decoding uses imageio's ffmpeg plugin (imageio-ffmpeg ships its own ffmpeg).
Nothing here imports Qt: progress is reported through a plain callback.
"""

import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image

from ..config import VIDEO_EXTS

SOURCE_FILE = "source_video.json"
FRAME_NAME = "frame_{:05d}.png"


ALL_FRAMES = {"mode": "all"}


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


def same_video(folder, video) -> bool:
    """True when `folder`'s frames were decoded from this very video file
    (same name, size and date) — with whatever sampling."""
    rec = read_source(folder)
    if not rec:
        return False
    cur = _source_record(Path(video))
    return all(rec.get(k) == cur[k] for k in ("name", "bytes", "mtime"))


def frames_from(folder, video, sampling=None) -> bool:
    """True when `folder`/frames already holds this very video's frames with
    this `sampling`, so decoding it again can be skipped."""
    if not same_video(folder, video):
        return False
    rec = read_source(folder)
    if normalize_sampling(rec.get("sampling")) != normalize_sampling(sampling):
        return False
    n = rec.get("frames") or 0
    first, last = rec.get("first", 0), rec.get("last", n - 1)
    frames = Path(folder) / "frames"
    return n > 0 and all((frames / FRAME_NAME.format(i)).is_file()
                         for i in (first, last))


def normalize_sampling(sampling) -> dict:
    """A sampling dict in its canonical form (missing / unknown = all frames)."""
    s = dict(sampling or ALL_FRAMES)
    mode = s.get("mode")
    if mode == "every" and int(s.get("step", 1)) > 1:
        return {"mode": "every", "step": int(s["step"])}
    if mode == "random" and int(s.get("count", 0)) > 0:
        return {"mode": "random", "count": int(s["count"]),
                "seed": int(s.get("seed", 0))}
    return dict(ALL_FRAMES)


def pick_frames(sampling, n_total):
    """The video frame numbers to keep, sorted — or None for all of them.
    Random picks are reproducible: the same seed gives the same frames."""
    s = normalize_sampling(sampling)
    if s["mode"] == "every":
        return list(range(0, n_total, s["step"])) if n_total else None
    if s["mode"] == "random":
        count = min(s["count"], n_total)
        return sorted(random.Random(s["seed"]).sample(range(n_total), count))
    return None


def kept_fps(sampling, video_fps):
    """Frame rate of the kept frames: the video's, fps / step for every n-th
    frame, None for random frames (unevenly spaced in time)."""
    s = normalize_sampling(sampling)
    if not video_fps:
        return None
    if s["mode"] == "every":
        return video_fps / s["step"]
    if s["mode"] == "random":
        return None
    return video_fps


def _open(video):
    import imageio.v2 as imageio            # raised to the caller if missing
    return imageio.get_reader(str(video), format="FFMPEG")


def video_info(video) -> dict:
    """What the file header says: fps (or None), estimated frame count (0 if
    unknown), duration in s (or None) and size (width, height) (or None).
    Cheap: nothing is decoded."""
    reader = _open(video)
    try:
        meta = reader.get_meta_data()
    finally:
        reader.close()
    fps = meta.get("fps") or None
    duration = meta.get("duration") or None
    n = meta.get("nframes")
    if not isinstance(n, (int, float)) or n != n or n in (float("inf"), 0):
        n = round(duration * fps) if fps and duration else 0
    size = meta.get("size") or meta.get("source_size")
    return {"fps": float(fps) if fps else None, "frames": int(n),
            "duration": float(duration) if duration else None,
            "size": tuple(size) if size else None}


def video_meta(video) -> tuple[float | None, int]:
    """(frames per second or None, estimated frame count or 0 if unknown)."""
    info = video_info(video)
    return info["fps"], info["frames"]


def count_frames(video) -> int:
    """The exact number of frames (ffmpeg reads the whole file, without
    writing anything; a few seconds for a long video). Falls back to the
    header's estimate."""
    try:
        import imageio_ffmpeg
        return int(imageio_ffmpeg.count_frames_and_secs(str(video))[0])
    except Exception as err:                 # noqa: BLE001 - estimate instead
        print(f"[video] could not count frames exactly ({err}); using the estimate")
        return video_meta(video)[1]


def extract_frames(video, out_dir, progress=None, sampling=None):
    """Decode `video` into `out_dir`/frames/ as PNGs named by their frame
    number in the video, keeping the frames `sampling` asks for (all of them
    by default; see the module docstring).

    `progress(done, total)` is called as frames are decoded (total is the
    number of frames that have to be read, an estimate when keeping every
    frame, 0 when unknown) and may return True to stop: the half-written
    frames are then removed and VideoCancelled raised. Frames are written to a
    temporary folder and only renamed to frames/ once complete, so a stopped or
    failed decode never leaves a folder that looks finished.
    Returns (number of frames kept, fps of the kept frames or None)."""
    video, out_dir = Path(video), Path(out_dir)
    sampling = normalize_sampling(sampling)
    video_fps, total = video_meta(video)
    keep = None                              # random: the frame numbers to keep
    n_video = None                           # exact frame count, when counted
    if sampling["mode"] == "random":
        n_video = count_frames(video)
        keep = pick_frames(sampling, n_video)
        total = keep[-1] + 1 if keep else total   # no need to read further
    step = sampling.get("step", 1)

    def wanted(i):
        if keep is not None:
            return i in keep_set
        return i % step == 0                 # all frames: step 1
    keep_set = set(keep or ())
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "frames.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    n = 0
    kept = []
    reader = _open(video)
    try:
        for i, frame in enumerate(reader):
            if wanted(i):
                Image.fromarray(frame).save(tmp / FRAME_NAME.format(i),
                                            compress_level=1)   # lossless, fast
                kept.append(i)
            n = i + 1
            if progress is not None and progress(n, max(total, n)):
                raise VideoCancelled(f"stopped after {n} frames")
            if keep and i >= keep[-1]:
                break
    except BaseException:
        reader.close()
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    reader.close()
    if not kept:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"no frames could be read from {video.name}")
    fps = kept_fps(sampling, video_fps)
    frames = out_dir / "frames"
    if frames.exists():                      # stale frames of an unsaved session
        shutil.rmtree(frames)
    os.replace(tmp, frames)
    (out_dir / SOURCE_FILE).write_text(json.dumps(
        {**_source_record(video), "sampling": sampling, "frames": len(kept),
         "first": kept[0], "last": kept[-1], "video_frames": n_video or n,
         "video_fps": video_fps, "fps": fps}, indent=2))
    print(f"[video] {video.name}: kept {len(kept)} of {n_video or n} frames "
          f"({sampling})"
          + (f" at {fps:g} fps" if fps else "") + f" -> {frames}")
    return len(kept), fps
