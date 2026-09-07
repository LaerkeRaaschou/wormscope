"""Keep more than one frame's image embedding around.

SAM2 encodes a frame with the image encoder (the expensive part) and caches the
result in ``inference_state["cached_features"]`` — but that line is::

    inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

i.e. it *replaces* the dict, so exactly ONE frame is remembered. Repeated clicks
on the same frame are therefore cheap, but every time you step away and come
back the frame is re-encoded from scratch.

:class:`FeatureCache` wraps ``predictor._get_image_feature`` and seeds that
one-slot cache from an LRU of its own, so revisiting a recent frame is free. It
also times every real encode, which is what makes the cost visible in the log.

Deliberately free of torch and Qt imports: it only ever moves opaque objects
around, which keeps it unit-testable without a model.
"""

from __future__ import annotations

import time
from collections import OrderedDict


def _tensor_bytes(obj, seen=None):
    """Rough byte size of a tensor / list / dict-of-tensors, duck-typed so no
    torch import is needed."""
    if obj is None:
        return 0
    numel = getattr(obj, "numel", None)
    element_size = getattr(obj, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel()) * int(element_size())
        except Exception:                                  # pragma: no cover
            return 0
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


class FeatureCache:
    """LRU over per-frame image embeddings, installed onto a SAM2 predictor.

    ``max_frames=None`` means unlimited — only use that deliberately (the
    "pre-encode everything" button), since a long clip can run to many GB.
    """

    def __init__(self, predictor, state, max_frames=16, log=print):
        self.predictor = predictor
        self.state = state
        self.max_frames = max_frames
        self.log = log
        self._store: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.seconds = 0.0
        self._inner = predictor._get_image_feature
        # An instance attribute shadows the class method, so SAM2's internal
        # `self._get_image_feature(...)` calls land here.
        predictor._get_image_feature = self._get_image_feature

    # --- the wrapper ---------------------------------------------------------
    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        entry = self._store.get(frame_idx)
        if entry is not None:
            # Hand SAM2's own one-slot cache the entry it is about to look for.
            inference_state["cached_features"] = {frame_idx: entry}
            self._store.move_to_end(frame_idx)
            self.hits += 1
            return self._inner(inference_state, frame_idx, batch_size)

        t0 = time.perf_counter()
        out = self._inner(inference_state, frame_idx, batch_size)
        dt = time.perf_counter() - t0
        self.misses += 1
        self.seconds += dt
        fresh = inference_state.get("cached_features", {}).get(frame_idx)
        if fresh is not None:
            self._store[frame_idx] = fresh
            self._trim()
        self.log(f"[sam2] embed frame {frame_idx}: {dt:.2f}s "
                 f"(miss, cache {len(self._store)}/{self.max_frames or '∞'})")
        return out

    def _trim(self):
        if self.max_frames is None:
            return
        while len(self._store) > max(1, self.max_frames):
            self._store.popitem(last=False)

    # --- introspection -------------------------------------------------------
    def __contains__(self, frame_idx):
        return frame_idx in self._store

    def __len__(self):
        return len(self._store)

    def bytes_per_frame(self):
        """Measured size of one cached embedding, or 0 if nothing is cached."""
        if not self._store:
            return 0
        first = next(iter(self._store.values()))
        return _tensor_bytes(first)

    def total_bytes(self):
        return self.bytes_per_frame() * len(self._store)

    def resize(self, max_frames):
        self.max_frames = max_frames
        self._trim()

    def clear(self):
        """Drop every remembered embedding — including the one slot SAM2 keeps
        itself, otherwise that frame silently survives the clear (and its
        memory with it)."""
        self._store.clear()
        try:
            self.state["cached_features"] = {}
        except TypeError:                                  # pragma: no cover
            pass

    def stats(self):
        total = self.hits + self.misses
        return {
            "frames_cached": len(self._store),
            "max_frames": self.max_frames,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "encode_seconds": round(self.seconds, 2),
            "mb_per_frame": round(self.bytes_per_frame() / 1e6, 1),
            "mb_total": round(self.total_bytes() / 1e6, 1),
        }

    def uninstall(self):
        """Put the predictor back the way it was (used by tests)."""
        try:
            del self.predictor._get_image_feature
        except AttributeError:                             # pragma: no cover
            self.predictor._get_image_feature = self._inner
        self.clear()
