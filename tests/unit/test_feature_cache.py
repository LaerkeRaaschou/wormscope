"""The image-embedding LRU.

SAM2 keeps exactly one frame's embedding, because its own cache line *replaces*
the dict rather than adding to it. Step away from a frame and back, and the
image encoder — the slow part of a click — runs again. FeatureCache wraps the
predictor and seeds that one slot from an LRU of its own.

Tested against a fake predictor that behaves the same way, so none of this
needs torch or a checkpoint.
"""

import pytest

from wormscope.segmentation.feature_cache import FeatureCache


class FakeTensor:
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n

    def element_size(self):
        return 4


class FakePredictor:
    """Mimics SAM2: a one-slot cache that is REPLACED on every miss."""

    def __init__(self):
        self.encodes = []

    def _get_image_feature(self, state, frame_idx, batch_size):
        _image, backbone = state["cached_features"].get(frame_idx, (None, None))
        if backbone is None:
            self.encodes.append(frame_idx)
            image, backbone = FakeTensor(10), {"fpn": [FakeTensor(1000)]}
            state["cached_features"] = {frame_idx: (image, backbone)}
        return ("features", frame_idx)


@pytest.fixture
def cache_setup():
    def _make(max_frames=3):
        pred = FakePredictor()
        state = {"cached_features": {}}
        cache = FeatureCache(pred, state, max_frames=max_frames,
                             log=lambda *_: None)
        return pred, state, cache
    return _make


def test_without_the_cache_sam2_re_encodes_every_revisit():
    """The behaviour being fixed — worth pinning so the fix has a baseline."""
    pred = FakePredictor()
    state = {"cached_features": {}}
    for f in (0, 1, 0, 1):
        pred._get_image_feature(state, f, 1)
    assert pred.encodes == [0, 1, 0, 1]


def test_a_revisited_frame_is_not_re_encoded(cache_setup):
    pred, state, _cache = cache_setup()
    for f in (0, 1, 0, 2, 1, 0):
        pred._get_image_feature(state, f, 1)
    assert pred.encodes == [0, 1, 2], "each frame encoded exactly once"


def test_hits_and_misses_are_counted(cache_setup):
    pred, state, cache = cache_setup()
    for f in (0, 0, 1, 0):
        pred._get_image_feature(state, f, 1)
    assert (cache.misses, cache.hits) == (2, 2)


def test_the_least_recently_used_frame_is_the_one_dropped(cache_setup):
    pred, state, cache = cache_setup(max_frames=2)
    for f in (0, 1):
        pred._get_image_feature(state, f, 1)
    pred._get_image_feature(state, 0, 1)         # 0 is now the recent one
    pred._get_image_feature(state, 2, 1)         # evicts 1, not 0
    assert 0 in cache and 2 in cache and 1 not in cache


def test_the_cache_never_grows_past_its_limit(cache_setup):
    pred, state, cache = cache_setup(max_frames=3)
    for f in range(10):
        pred._get_image_feature(state, f, 1)
    assert len(cache) == 3


def test_an_evicted_frame_has_to_be_encoded_again(cache_setup):
    pred, state, _cache = cache_setup(max_frames=2)
    for f in (0, 1, 2, 0):
        pred._get_image_feature(state, f, 1)
    assert pred.encodes == [0, 1, 2, 0]


def test_the_per_frame_size_is_measured_from_a_real_entry(cache_setup):
    """The estimate has to come from a measurement, not a guess: it is what a
    memory warning would be based on."""
    pred, state, cache = cache_setup()
    pred._get_image_feature(state, 0, 1)
    assert cache.bytes_per_frame() == (10 + 1000) * 4
    assert cache.total_bytes() == cache.bytes_per_frame()


def test_an_unlimited_cache_keeps_everything(cache_setup):
    pred, state, cache = cache_setup(max_frames=None)
    for f in range(20):
        pred._get_image_feature(state, f, 1)
    for f in range(20):
        pred._get_image_feature(state, f, 1)
    assert pred.encodes == list(range(20)), "no frame encoded twice"
    assert len(cache) == 20


def test_resize_shrinks_an_existing_cache(cache_setup):
    pred, state, cache = cache_setup(max_frames=None)
    for f in range(6):
        pred._get_image_feature(state, f, 1)
    cache.resize(2)
    assert len(cache) == 2


def test_clearing_empties_it(cache_setup):
    pred, state, cache = cache_setup()
    pred._get_image_feature(state, 0, 1)
    cache.clear()
    assert len(cache) == 0


def test_uninstall_puts_the_predictor_back_as_it_was(cache_setup):
    """The wrapper is installed by assignment onto the instance; leaving it in
    place after the backend is dropped would keep the whole cache alive."""
    pred, _state, cache = cache_setup()
    cache.uninstall()
    assert pred._get_image_feature.__self__ is pred
