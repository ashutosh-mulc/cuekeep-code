"""Paging policy, fetch/commit protocol and cache bookkeeping, on CPU."""
import math

import pytest

torch = pytest.importorskip("torch")

from cuekeep.baselines import random_regions  # noqa: E402
from cuekeep.cache import (  # noqa: E402
    PagedVisualCache, SlotPool, attend_full, attend_mixed, region_ids_for_grid, region_ids_for_spans,
)


# ----------------------------------------------------------------------------- SlotPool

def test_slot_pool_never_exceeds_capacity_and_respects_min_residency():
    pool = SlotPool(capacity=3, fetch_budget=2, min_residency=2)
    gen = torch.Generator().manual_seed(0)
    for step in range(50):
        before = dict(pool.fetched_at)
        _, evictions = pool.plan(torch.rand(10, generator=gen), step)
        assert len(pool.region_to_slot) <= 3
        assert len(set(pool.region_to_slot.values())) == len(pool.region_to_slot)   # no shared slots
        for r, _ in evictions:
            assert step - before[r] >= 2


def test_slot_pool_fetches_top_scored_first():
    pool = SlotPool(capacity=4, fetch_budget=2)
    fetches, _ = pool.plan(torch.tensor([0.1, 5.0, 0.2, 4.0, 0.3]), step=0)
    assert {r for r, _ in fetches} == {1, 3}


def test_slot_pool_never_evicts_a_region_fetched_in_the_same_step():
    # Regression (review 2026-11-12): with min_residency=0 the old code fetched region 2, then evicted it
    # in the same plan() call for the lower-scored region 3.
    pool = SlotPool(capacity=1, fetch_budget=2, min_residency=0)
    pool.plan(torch.tensor([0.0, 0.0, 0.0, 0.0]), 0)                 # region 0 takes the only slot
    fetches, evictions = pool.plan(torch.tensor([0.0, 0.0, 5.0, 4.0]), 1)
    assert fetches == [(2, 0)] and evictions == [(0, 0)]
    assert pool.region_to_slot == {2: 0}


def test_slot_pool_admission_compares_current_scores():
    pool = SlotPool(capacity=1, fetch_budget=1, min_residency=0, admit_margin=0.5)
    pool.plan(torch.tensor([3.0, 0.0]), 0)                           # region 0 resident
    f, e = pool.plan(torch.tensor([3.0, 3.2]), 1)                    # 3.2 does not beat 3.0 + 0.5
    assert f == [] and e == []
    f, e = pool.plan(torch.tensor([3.0, 3.6]), 2)
    assert f == [(1, 0)] and e == [(0, 0)]


# ----------------------------------------------------------------------------- cache

def _cache(R_cap, h=4, w=4, g=2, kvh=2, d=8, lookahead=1, **cfg):
    rid = region_ids_for_grid(h, w, g)
    gen = torch.Generator().manual_seed(0)
    k = torch.randn(kvh, h * w, d, generator=gen)
    v = torch.randn(kvh, h * w, d, generator=gen)
    cache = PagedVisualCache({"slot_capacity": R_cap, "fetch_budget": R_cap, "min_residency": 0,
                              "lookahead": lookahead, **cfg}, device="cpu")
    cache.ingest(0, k, v, rid)
    return cache, k, v, rid, gen


def _attend_from_cache(cache, q, kvh, d):
    vk, vv, bias = cache.attention_inputs(0)
    qk = q.reshape(kvh, -1, d)
    return torch.softmax((qk @ vk.transpose(-1, -2)) * d ** -0.5 + bias, -1) @ vv


def test_cache_reproduces_full_attention_when_all_committed():
    cache, k, v, rid, gen = _cache(R_cap=4, lookahead=0)
    kvh, d = 2, 8
    q = torch.randn(kvh * 2, 1, d, generator=gen)                    # GQA: 2 query heads per KV head
    cache.step(0, q, step=0)
    assert cache.commit(0, step=0) and bool(cache.layers[0].committed.all())
    assert cache.bytes_h2d == 2 * k.numel() * k.element_size()
    out = _attend_from_cache(cache, q, kvh, d)
    torch.testing.assert_close(out, attend_full(q.reshape(kvh, 2, d), k, v, d ** -0.5), rtol=1e-5, atol=1e-5)


def test_fetch_is_invisible_until_committed_at_lookahead():
    cache, k, v, rid, gen = _cache(R_cap=1, lookahead=1)
    st = cache.layers[0]
    q = torch.randn(4, 1, 8, generator=gen)
    cache.step(0, q, step=5)
    assert len(st.pending) == 1 and not st.committed.any() and int(st.slot_len.sum()) == 0
    assert cache.commit(0, step=5) == []                              # not due yet
    (region,) = cache.commit(0, step=6)
    assert bool(st.committed[region]) and int(st.slot_len[0]) == 4
    # Before commit, attention uses only stand-ins; after, the committed region is exact.
    out = _attend_from_cache(cache, q, 2, 8)
    resident = torch.zeros(4, dtype=torch.bool)
    resident[region] = True
    ref = attend_mixed(q.reshape(2, 2, 8), None, None, k, v, rid, st.pooled, resident, 8 ** -0.5)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_eviction_restores_standin_immediately_and_drops_pending():
    cache, k, v, rid, gen = _cache(R_cap=1, lookahead=1)
    st, pool = cache.layers[0], cache.pools[0]
    big = lambda r: torch.nn.functional.one_hot(torch.tensor(r), 4).float() * 10      # noqa: E731
    pool.plan = (lambda orig: (lambda s, t: orig(big(t % 4), t)))(pool.plan)           # force region = step % 4
    q = torch.randn(4, 1, 8, generator=gen)
    cache.step(0, q, 0)
    cache.commit(0, 1)
    assert bool(st.committed[0])
    cache.step(0, q, 1)                                    # evicts region 0, fetches region 1 into slot 0
    assert not st.committed[0] and int(st.slot_len[0]) == 0
    cache.step(0, q, 2)                                    # region 1 still pending; evicted before commit
    assert [p.region for p in st.pending] == [2]
    cache.commit(0, 3)
    assert st.committed.tolist() == [False, False, True, False]


def test_host_store_is_region_contiguous():
    cache, k, v, rid, _ = _cache(R_cap=1)
    st = cache.layers[0]
    for r in range(4):
        a, m = st.region_start[r], st.region_len[r]
        torch.testing.assert_close(st.host_k[a:a + m], k[:, rid == r].permute(1, 0, 2))
        assert st.host_k[a:a + m].is_contiguous()


def test_region_scores_use_per_head_shares():
    # Two loud heads attend almost only to text; two quiet heads attend strongly to region 3.
    # Raw max of log-mass would follow the loud heads to region 1. Per-head shares must pick region 3.
    # Keys are built deterministically: every token of region r has key 3·e_r, so no seed can change the answer.
    h, w, g, kvh, d = 4, 4, 2, 4, 8
    rid = region_ids_for_grid(h, w, g)
    k = torch.zeros(kvh, h * w, d)
    for r in range(4):
        k[:, rid == r, r] = 3.0
    cache = PagedVisualCache({"slot_capacity": 1}, device="cpu")
    cache.ingest(0, k, k.clone(), rid)
    q = torch.zeros(4, 1, d)
    q[:2, 0, 1] = 50.0                                     # loud heads point at region 1
    q[2:, 0, 3] = 6.0                                      # quiet heads point at region 3
    other = torch.tensor([[60.0], [60.0], [-10.0], [-10.0]])          # loud heads' text logits dominate
    agg = cache.region_scores(0, q, other_lse=other)
    assert int(agg.argmax()) == 3
    from cuekeep.cache import demand_scores
    raw = demand_scores(q, cache.layers[0].pooled, d ** -0.5).amax(dim=(0, 1))
    assert int(raw.argmax()) == 1                          # the old aggregation gets this wrong


def test_ingest_rejects_second_call_for_same_layer():
    cache, k, v, rid, _ = _cache(R_cap=1)
    with pytest.raises(ValueError):
        cache.ingest(0, k, v, rid)


def test_multi_image_ingest_with_span_ids():
    rid = region_ids_for_spans([(4, 4), (2, 4)], g=2)
    k = torch.randn(2, 24, 8)
    cache = PagedVisualCache({"slot_capacity": 2}, device="cpu")
    cache.ingest(0, k, k.clone(), rid)
    assert cache.layers[0].pooled.num_regions == 6


def test_random_regions_budget_and_seed():
    a = random_regions(20, 5, torch.Generator().manual_seed(3))
    b = random_regions(20, 5, torch.Generator().manual_seed(3))
    assert int(a.sum()) == 5 and torch.equal(a, b)
