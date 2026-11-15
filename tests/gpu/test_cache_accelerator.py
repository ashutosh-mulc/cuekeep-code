"""PagedVisualCache on an accelerator: CUDA on the server (streams, events, pinned host memory), and MPS on
a Mac (device placement only; the stream/event path is CUDA-only). Skipped when neither is available.

    pytest -q -m gpu
"""
import pytest

torch = pytest.importorskip("torch")

from cuekeep.cache import PagedVisualCache, attend_full, attend_mixed, region_ids_for_grid  # noqa: E402

pytestmark = pytest.mark.gpu
DEVICES = [d for d, ok in (("cuda", torch.cuda.is_available()),
                           ("mps", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())) if ok]
if not DEVICES:
    pytest.skip("no CUDA or MPS device", allow_module_level=True)


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def _cache(dev, R_cap, lookahead, h=4, w=4, g=2, kvh=2, d=8):
    rid = region_ids_for_grid(h, w, g)
    gen = torch.Generator().manual_seed(0)
    k = torch.randn(kvh, h * w, d, generator=gen).to(dev)
    v = torch.randn(kvh, h * w, d, generator=gen).to(dev)
    cache = PagedVisualCache({"slot_capacity": R_cap, "fetch_budget": R_cap, "min_residency": 0,
                              "lookahead": lookahead}, device=dev)
    cache.ingest(0, k, v, rid)
    return cache, k, v, rid, gen


def _attend(cache, q, kvh, d):
    vk, vv, bias = cache.attention_inputs(0)
    qk = q.reshape(kvh, -1, d)
    return torch.softmax((qk @ vk.transpose(-1, -2)) * d ** -0.5 + bias, -1) @ vv


@pytest.mark.parametrize("dev", DEVICES)
def test_all_committed_equals_full_attention(dev):
    cache, k, v, rid, gen = _cache(dev, R_cap=4, lookahead=0)
    q = torch.randn(4, 1, 8, generator=gen).to(dev)
    cache.step(0, q, step=0)
    assert cache.commit(0, step=0)
    _sync(dev)
    st = cache.layers[0]
    assert bool(st.committed.all()) and st.slot_k.device.type == dev
    if dev == "cuda":
        assert st.host_k.is_pinned() and st.host_v.is_pinned()
    out = _attend(cache, q, 2, 8)
    torch.testing.assert_close(out.cpu(), attend_full(q.reshape(2, 2, 8), k, v, 8 ** -0.5).cpu(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dev", DEVICES)
def test_fetch_invisible_until_commit_then_exact(dev):
    cache, k, v, rid, gen = _cache(dev, R_cap=1, lookahead=1)
    st = cache.layers[0]
    q = torch.randn(4, 1, 8, generator=gen).to(dev)
    cache.step(0, q, step=5)
    assert len(st.pending) == 1 and not bool(st.committed.any())
    assert cache.commit(0, step=5) == []
    (region,) = cache.commit(0, step=6)
    _sync(dev)
    assert bool(st.committed[region]) and int(st.slot_len[0]) == 4
    resident = torch.zeros(4, dtype=torch.bool)
    resident[region] = True
    ref = attend_mixed(q.reshape(2, 2, 8).cpu(), None, None, k.cpu(), v.cpu(), rid, st.pooled.__class__(
        st.pooled.keys.cpu(), st.pooled.values.cpu(), st.pooled.counts.cpu(), st.pooled.log_m.cpu()), resident, 8 ** -0.5)
    torch.testing.assert_close(_attend(cache, q, 2, 8).cpu(), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dev", DEVICES)
def test_eviction_reuses_slot_after_queued_reads(dev):
    """Churn through every region with capacity 1; each committed slot must hold that region's exact KV."""
    cache, k, v, rid, gen = _cache(dev, R_cap=1, lookahead=1)
    st, pool = cache.layers[0], cache.pools[0]
    target = lambda r: torch.nn.functional.one_hot(torch.tensor(r), 4).float() * 10      # noqa: E731
    pool.plan = (lambda orig: (lambda s, t: orig(target(t % 4), t)))(pool.plan)
    q = torch.randn(4, 1, 8, generator=gen).to(dev)
    for t in range(12):
        cache.commit(0, t)
        _attend(cache, q, 2, 8)                       # queue a read of the slots before they are reused
        cache.step(0, q, t)
    cache.commit(0, 12)
    _sync(dev)
    (r,) = st.committed.nonzero().flatten().tolist()
    m = int(st.slot_len[0])
    torch.testing.assert_close(st.slot_k[0, :m].cpu(), k[:, rid == r].permute(1, 0, 2).cpu())
