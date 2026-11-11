"""Paging policy and cache bookkeeping on CPU."""
import pytest

torch = pytest.importorskip("torch")

from cuekeep.baselines import random_regions  # noqa: E402
from cuekeep.cache import PagedVisualCache, SlotPool, attend_full, region_ids_for_grid  # noqa: E402


def test_slot_pool_never_exceeds_capacity_and_respects_min_residency():
    pool = SlotPool(capacity=3, fetch_budget=2, min_residency=2)
    gen = torch.Generator().manual_seed(0)
    for step in range(50):
        before = dict(pool.fetched_at)
        fetches, evicted = pool.plan(torch.rand(10, generator=gen), step)
        assert len(pool.region_to_slot) <= 3
        assert len(set(pool.region_to_slot.values())) == len(pool.region_to_slot)   # no shared slots
        for r in evicted:
            assert step - before[r] >= 2


def test_slot_pool_fetches_top_scored_first():
    pool = SlotPool(capacity=4, fetch_budget=2)
    fetches, _ = pool.plan(torch.tensor([0.1, 5.0, 0.2, 4.0, 0.3]), step=0)
    assert {r for r, _ in fetches} == {1, 3}


def test_cache_attention_inputs_reproduce_full_attention_when_all_resident():
    h, w, g, kvh, d = 4, 4, 2, 2, 8
    rid = region_ids_for_grid(h, w, g)
    R = int(rid.max()) + 1
    gen = torch.Generator().manual_seed(0)
    k = torch.randn(kvh, h * w, d, generator=gen)
    v = torch.randn(kvh, h * w, d, generator=gen)
    cache = PagedVisualCache({"slot_capacity": R, "fetch_budget": R, "min_residency": 0}, device="cpu")
    cache.ingest(0, k, v, rid)
    q = torch.randn(kvh * 2, 1, d, generator=gen)        # GQA: 2 query heads per KV head
    cache.step(0, q, step=0)
    assert cache.pools[0].resident_mask(R).all()
    assert cache.bytes_h2d == 2 * k.numel() * k.element_size()

    vk, vv, bias = cache.attention_inputs(0)
    qk = q.reshape(kvh, 2, d)
    logits = (qk @ vk.transpose(-1, -2)) * d ** -0.5 + bias
    out = torch.softmax(logits, -1) @ vv
    torch.testing.assert_close(out, attend_full(qk, k, v, d ** -0.5), rtol=1e-5, atol=1e-5)


def test_random_regions_budget_and_seed():
    a = random_regions(20, 5, torch.Generator().manual_seed(3))
    b = random_regions(20, 5, torch.Generator().manual_seed(3))
    assert int(a.sum()) == 5 and torch.equal(a, b)
