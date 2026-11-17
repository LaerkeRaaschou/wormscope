"""SAM2 interactive segmentation (video predictor, lazily loaded).

The config + checkpoint are resolved by :mod:`wormscope.settings`: normally the
config ships inside the sam2 package and the checkpoint is downloaded once
(:mod:`wormscope.segmentation.sam2_models`); hand-picked files override that.

To plug in a different segmenter, implement the same three methods
(``add_points``, ``propagate``, ``reset``) and point
:data:`wormscope.segmentation.SEGMENTER_FACTORY` at your class.
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config, settings
from .feature_cache import FeatureCache


class MissingSam2Paths(RuntimeError):
    """Raised when no usable SAM2 config/checkpoint could be found, so the GUI
    can offer a file picker instead of a dead-end error."""


def sam2_import_problem():
    """Why `import sam2` doesn't give us the real SAM2 package, or "" if it does.

    The classic trap: starting wormscope from a folder that contains a *clone*
    of the SAM2 repo called ``sam2``. That folder has no __init__.py, so Python
    imports it as an empty namespace package (``sam2.__file__`` is None) and
    the real package — installed or not — is never seen."""
    try:
        import sam2
    except Exception as err:                 # noqa: BLE001 - shown to the user
        return (f"SAM2 is not installed in this Python environment ({err}).\n"
                f"Python in use: {sys.executable}")
    if getattr(sam2, "__file__", None) is None:
        where = ", ".join(str(p) for p in getattr(sam2, "__path__", []))
        return ("Python found a plain folder named 'sam2' instead of the SAM2 "
                f"package:\n{where}\n\nStart wormscope from another folder (or "
                "move that folder), and make sure SAM2 is installed in this "
                f"environment.\nPython in use: {sys.executable}")
    return ""


def sam2_package_dir():
    """Folder of the installed `sam2` package, or None if it isn't usable."""
    if sam2_import_problem():
        return None
    import sam2
    return Path(sam2.__file__).resolve().parent


def sam2_config_dir():
    """The installed package's configs/ folder — a good place to start a file
    dialog. None if sam2 isn't installed or ships no configs/."""
    pkg = sam2_package_dir()
    if pkg is None:
        return None
    cfg_dir = pkg / "configs"
    return cfg_dir if cfg_dir.is_dir() else pkg


def hydra_config_name(cfg_path):
    """Turn whatever the user picked into something Hydra can actually load.

    SAM2 composes its config with Hydra, whose search path is the *installed
    sam2 package* (``pkg://sam2``). Hydra therefore expects a package-relative
    NAME such as ``configs/sam2.1/sam2.1_hiera_l.yaml`` — hand it an absolute
    path and it strips the leading slash and fails with
    "Cannot find primary config 'Users/…'".

    Returns ``(config_name, extra_hydra_overrides)``:

    * a name that is already relative is passed straight through;
    * a file inside the sam2 package becomes its package-relative name;
    * a file anywhere else keeps its bare name, and its folder is added to
      Hydra's search path via an override.
    """
    raw = str(cfg_path)
    p = Path(raw).expanduser()
    if not p.is_absolute() and not p.exists():
        return raw, []                       # already a Hydra config name
    p = p.resolve()
    pkg = sam2_package_dir()
    if pkg is not None:
        try:
            return str(p.relative_to(pkg)), []
        except ValueError:
            pass                             # not inside the package
    return p.name, [f"hydra.searchpath=[file://{p.parent}]"]


# Checkpoint filename fragment -> config filename fragment, e.g.
# sam2.1_hiera_large.pt  goes with  configs/sam2.1/sam2.1_hiera_l.yaml
_SIZES = {"hiera_tiny": "hiera_t", "hiera_small": "hiera_s",
          "hiera_base_plus": "hiera_b+", "hiera_large": "hiera_l"}


def pair_warning(config_name, ckpt_path):
    """Warn when the config and checkpoint are obviously not a matching pair —
    that mismatch otherwise surfaces as a wall of missing state-dict keys.

    Returns a human-readable string, or "" when they look consistent."""
    cfg = str(config_name).lower()
    ckpt = Path(ckpt_path).name.lower()
    problems = []
    cfg_21 = "sam2.1" in cfg
    ckpt_21 = "sam2.1" in ckpt or "sam2_1" in ckpt
    if cfg_21 != ckpt_21:
        problems.append(f"the config looks like SAM {'2.1' if cfg_21 else '2.0'} "
                        f"but the checkpoint looks like SAM "
                        f"{'2.1' if ckpt_21 else '2.0'}")
    for ckpt_frag, cfg_frag in _SIZES.items():
        if ckpt_frag in ckpt and cfg_frag not in cfg:
            problems.append(f"the checkpoint is '{ckpt_frag}' but the config is "
                            f"not '{cfg_frag}'")
            break
    if not problems:
        return ""
    return ("Config and checkpoint may not match: " + "; ".join(problems)
            + ". Loading will fail with missing/unexpected keys if so.")


# Everything a session owns privately: prompts, memory, object bookkeeping.
# Any other key in an inference state (the decoded frames, the device, the
# frame size) is shared between sessions. Old SAM2 keys are listed too so this
# keeps working across versions.
_PER_SESSION_KEYS = frozenset((
    "point_inputs_per_obj", "mask_inputs_per_obj", "cached_features",
    "constants", "obj_id_to_idx", "obj_idx_to_id", "obj_ids",
    "output_dict_per_obj", "temp_output_dict_per_obj", "frames_tracked_per_obj",
    "output_dict", "temp_output_dict", "consolidated_frame_inds",
    "tracking_has_started",
))


def prepare_jpeg_dir(frames):
    """SAM2's video predictor wants a folder of numbered JPEGs. Write the
    frames as 0.jpg, 1.jpg, … in frame order so SAM frame_idx == our index."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="wormscope_sam_"))
    for i in range(len(frames)):
        Image.fromarray(frames.get(i)).save(d / f"{i}.jpg", quality=95)
    return d


class Sam2Backend:
    """Thin wrapper over SAM2's video predictor. Imported lazily so the app
    runs without torch/sam2 installed. Point prompts on any frame drive a real
    per-frame segmentation of that object across the whole clip (propagation),
    not a copied mask.

    Requires the `sam2` package plus a checkpoint + config. The paths are
    resolved once by wormscope.settings (saved file -> SAM2_CFG/SAM2_CKPT env
    vars -> config.py), so they never have to be re-entered."""

    def __init__(self, frames, shape, cfg=None, ckpt=None, device=None):
        # Resolve the paths BEFORE importing torch, so a missing config is
        # reported as MissingSam2Paths (which the GUI answers with a file
        # picker) rather than as an unrelated ImportError.
        # A broken install is reported first: no point downloading a model
        # that can't be loaded.
        problem = sam2_import_problem()
        if problem:
            raise RuntimeError(problem)
        cfg, ckpt, source = settings.sam2_paths(cfg, ckpt)
        if not cfg or not ckpt:
            raise MissingSam2Paths(
                "wormscope needs a SAM2 checkpoint — download the model, or "
                "choose a config + checkpoint by hand.")
        print(f"[sam2] {source}")
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        device = device or settings.sam2_device() or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")
        self._torch = torch
        # Hydra wants a config NAME, not a path — see hydra_config_name().
        config_name, overrides = hydra_config_name(cfg)
        print(f"[sam2] config '{config_name}'"
              + (f" (+ {overrides})" if overrides else ""))
        warning = pair_warning(config_name, ckpt)
        if warning:
            print(f"[sam2] {warning}")
        extra = {"hydra_overrides_extra": overrides} if overrides else {}
        self.predictor = build_sam2_video_predictor(config_name, ckpt,
                                                    device=device, **extra)
        self.frames_dir = prepare_jpeg_dir(frames)
        # One state is built the normal way; it decodes the whole clip and is
        # then used only as the template every per-worm session is cloned from.
        self._template = self.predictor.init_state(video_path=str(self.frames_dir))
        self.sessions = {}                   # obj_id -> its own inference_state
        self._prompted = set()               # obj_ids that have any prompt
        self.shape = shape
        self.n_frames = len(frames)
        # Remember more than SAM2's single frame of image features, so stepping
        # back to a frame you already clicked on doesn't re-run the encoder.
        # Shared by every session: an embedding depends on the frame and the
        # model, never on which worm is being tracked.
        self.features = FeatureCache(self.predictor, self._template,
                                     max_frames=config.SAM2_EMBED_CACHE)
        print(f"[sam2] ready on {device}; frames -> {self.frames_dir}")

    # --- one session per worm ------------------------------------------------
    def session(self, obj_id):
        """The inference state for this worm, created on first use.

        Each worm gets its own SAM2 session, so a prompt or a re-track on one
        worm cannot disturb another's memory or masks. The decoded frames are
        shared between sessions — they are the big allocation, and they are the
        same pixels for everyone."""
        state = self.sessions.get(int(obj_id))
        if state is None:
            state = self._clone_template()
            self.sessions[int(obj_id)] = state
            print(f"[sam2] new session for worm {obj_id} "
                  f"({len(self.sessions)} open)")
        return state

    def _clone_template(self):
        """A fresh, empty inference state that shares the template's frames.

        Anything per-session (prompts, memory, object bookkeeping) is replaced
        with a new empty container of the same type; everything else — the
        decoded frames above all — is shared by reference. Rebuilding it by type
        rather than by a hardcoded list keeps this working if SAM2 adds a key."""
        fresh = {}
        for key, value in self._template.items():
            if key in _PER_SESSION_KEYS:
                fresh[key] = type(value)()   # {} / OrderedDict() / [] / False
            else:
                fresh[key] = value           # images, device, sizes …
        return fresh

    def forget(self, obj_id):
        """Drop a worm's session (deleted worm, or its prompts were cleared)."""
        self.sessions.pop(int(obj_id), None)
        self._prompted.discard(int(obj_id))

    def prompted_objects(self):
        """Worm ids that have at least one prompt, in the order first prompted."""
        return [oid for oid in self.sessions if oid in self._prompted]

    def has_prompts(self, obj_id):
        return int(obj_id) in self._prompted

    def _mask_of(self, obj_id, ids, logits):
        for k, i in enumerate(ids):
            if int(i) == int(obj_id):
                return (logits[k, 0] > 0.0).detach().cpu().numpy()
        return None

    def add_points(self, frame_idx, obj_id, points, labels):
        """points: list of (x, y); labels: 1 = worm, 0 = not worm. Returns the
        object's mask on this frame (a preview) or None.

        The full point list is re-sent every time (SAM2 clears the previous
        points for this frame/object), so flipping or dropping a point is just a
        matter of calling this again with the corrected list."""
        t0 = time.perf_counter()
        _, ids, logits = self.predictor.add_new_points_or_box(
            inference_state=self.session(obj_id), frame_idx=int(frame_idx),
            obj_id=int(obj_id),
            points=np.asarray(points, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int32))
        self._prompted.add(int(obj_id))
        print(f"[sam2] add_points frame {frame_idx} obj {obj_id} "
              f"({len(points)} pts): {time.perf_counter() - t0:.2f}s")
        return self._mask_of(obj_id, ids, logits)

    def set_points(self, frame_idx, obj_id, points, labels):
        """Make this frame's point prompts exactly `points` — used by undo and
        redo. An empty list removes the frame's prompts altogether."""
        if points:
            return self.add_points(frame_idx, obj_id, points, labels)
        self.clear_frame(frame_idx, obj_id)
        return None

    def clear_frame(self, frame_idx, obj_id):
        """Remove every prompt (points or mask) this worm has on one frame. If
        that was its last prompt anywhere, its session is dropped entirely."""
        oid = int(obj_id)
        state = self.sessions.get(oid)
        if state is None or oid not in state["obj_id_to_idx"]:
            return
        self.predictor.clear_all_prompts_in_frame(
            state, int(frame_idx), oid, need_output=False)
        obj_idx = state["obj_id_to_idx"][oid]
        if not (state["point_inputs_per_obj"][obj_idx]
                or state["mask_inputs_per_obj"][obj_idx]):
            self.forget(oid)

    def add_mask(self, frame_idx, obj_id, mask):
        """Use an existing (e.g. brush-corrected) mask as the prompt for this
        frame — the mask equivalent of add_points. That frame becomes an anchor
        the tracker works from. Returns SAM2's own version of the mask."""
        t0 = time.perf_counter()
        _, ids, logits = self.predictor.add_new_mask(
            inference_state=self.session(obj_id), frame_idx=int(frame_idx),
            obj_id=int(obj_id), mask=np.asarray(mask, dtype=bool))
        self._prompted.add(int(obj_id))
        print(f"[sam2] add_mask frame {frame_idx} obj {obj_id}: "
              f"{time.perf_counter() - t0:.2f}s")
        return self._mask_of(obj_id, ids, logits)

    def propagate(self, obj_id, start_frame_idx=None, reverse=False,
                  max_frames=None):
        """Track ONE worm and yield (frame_idx, bool_mask) as it goes.

        Only that worm's session is touched, so nothing another worm has —
        prompts, memory or masks — can change as a side effect. Defaults to the
        whole clip from its earliest prompt; pass `start_frame_idx` to re-track
        from a frame you just corrected and `reverse=True` to run towards 0."""
        state = self.session(obj_id)
        kwargs = {}
        if start_frame_idx is not None:
            kwargs["start_frame_idx"] = int(start_frame_idx)
        if reverse:
            kwargs["reverse"] = True
        if max_frames is not None:
            kwargs["max_frame_num_to_track"] = int(max_frames)
        t0 = time.perf_counter()
        n = 0
        for frame_idx, ids, logits in self.predictor.propagate_in_video(
                state, **kwargs):
            mask = self._mask_of(obj_id, ids, logits)
            n += 1
            if mask is not None:
                yield int(frame_idx), mask
        print(f"[sam2] worm {obj_id}: propagated {n} frames "
              f"({'backwards' if reverse else 'forwards'}"
              f"{'' if start_frame_idx is None else f' from {start_frame_idx}'})"
              f" in {time.perf_counter() - t0:.1f}s")

    # --- image embeddings ----------------------------------------------------
    def prewarm(self, frame_indices, progress=None):
        """Encode these frames now so later clicks on them are instant.

        `progress(done, total, seconds, bytes_each)` is called after each frame
        and may return True to stop early. Returns the number encoded."""
        indices = [i for i in frame_indices if i not in self.features]
        total = len(indices)
        done = 0
        with self._torch.inference_mode():
            for idx in indices:
                t0 = time.perf_counter()
                self.predictor._get_image_feature(self._template, int(idx), 1)
                done += 1
                if progress is not None and progress(
                        done, total, time.perf_counter() - t0,
                        self.features.bytes_per_frame()):
                    break
        return done

    def prewarm_all(self, progress=None):
        """Encode every frame up front, keeping them all in the cache."""
        self.features.resize(None)               # unlimited: we want them kept
        return self.prewarm(range(self.n_frames), progress)

    def estimate_prewarm_bytes(self, n_frames=None):
        """Bytes the full pre-encode would need, from a measured frame. Encodes
        one frame first if the cache is still empty. 0 if unmeasurable."""
        if not len(self.features):
            with self._torch.inference_mode():
                self.predictor._get_image_feature(self._template, 0, 1)
        per = self.features.bytes_per_frame()
        return per * (self.n_frames if n_frames is None else n_frames)

    def cache_stats(self):
        return self.features.stats()

    def reset(self):
        """Forget every prompt for every worm. The frames and the embedding
        cache survive — they cost the most and nothing about them is stale."""
        self.sessions.clear()
        self._prompted.clear()
        self.features.resize(config.SAM2_EMBED_CACHE)
