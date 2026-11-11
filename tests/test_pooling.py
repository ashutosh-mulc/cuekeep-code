"""Correctness of the pooled stand-in math (research/method.md). These run on CPU."""
import math

import pytest

torch = pytest.importorskip("torch")

from cuekeep.cache import (  # noqa: E402
    attend_full, attend_mixed, demand_scores, exact_region_log_mass, pool_regions, pooled_logits,
    region_ids_for_grid,
)

D = 16
SCALE = D ** -0.5


def _setup(h=6, w=6, g=2, kvh=2, nq=3, spread=0.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rid = region_ids_for_grid(h, w, g)
    R = int(rid.max()) + 1
    centres = torch.randn(kvh, R, D, generator=gen, dtype=torch.float64)
    k = centres[:, rid, :] + spread * torch.randn(kvh, h * w, D, generator=gen, dtype=torch.float64)
    v = torch.randn(kvh, h * w, D, generator=gen, dtype=torch.float64)
    q = torch.randn(kvh, nq, D, generator=gen, dtype=torch.float64)
    return q, k, v, rid, R


def test_region_ids_tile_grid_with_ragged_edges():
    rid = region_ids_for_grid(5, 7, 2)
    assert rid.shape == (35,)
    counts = torch.bincount(rid)
    assert counts.numel() == math.ceil(5 / 2) * math.ceil(7 / 2)
    assert counts.max() == 4 and counts.min() == 1      # corner region is 1×1
    assert int(counts.sum()) == 35


def test_log_m_standin_exact_when_region_keys_identical():
    # Values differ within a region, keys do not. The stand-in must be exact.
    q, k, v, rid, R = _setup(spread=0.0)
    pooled = pool_regions(k, v, rid, R)
    out = attend_mixed(q, None, None, k, v, rid, pooled, torch.zeros(R, dtype=torch.bool), SCALE)
    torch.testing.assert_close(out, attend_full(q, k, v, SCALE))


def test_without_log_m_bias_standin_is_wrong():
    q, k, v, rid, R = _setup(h=6, w=4, g=3, spread=0.0)   # unequal region sizes, so the bias matters
    pooled = pool_regions(k, v, rid, R)
    out = attend_mixed(q, None, None, k, v, rid, pooled, torch.zeros(R, dtype=torch.bool), SCALE, use_log_m=False)
    assert not torch.allclose(out, attend_full(q, k, v, SCALE))


def test_all_resident_equals_full_attention():
    q, k, v, rid, R = _setup(spread=1.0)
    pooled = pool_regions(k, v, rid, R)
    out = attend_mixed(q, None, None, k, v, rid, pooled, torch.ones(R, dtype=torch.bool), SCALE)
    torch.testing.assert_close(out, attend_full(q, k, v, SCALE))


def test_mixed_with_dense_text_tokens_and_partial_residency():
    q, k, v, rid, R = _setup(spread=0.0)
    gen = torch.Generator().manual_seed(1)
    tk = torch.randn(2, 5, D, generator=gen, dtype=torch.float64)
    tv = torch.randn(2, 5, D, generator=gen, dtype=torch.float64)
    resident = torch.zeros(R, dtype=torch.bool)
    resident[::2] = True
    pooled = pool_regions(k, v, rid, R)
    out = attend_mixed(q, tk, tv, k, v, rid, pooled, resident, SCALE)
    ref = attend_full(q, torch.cat([tk, k], 1), torch.cat([tv, v], 1), SCALE)
    torch.testing.assert_close(out, ref)   # exact: identical keys inside regions


def test_jensen_pooled_logit_lower_bounds_true_log_mass():
    q, k, v, rid, R = _setup(spread=0.7)
    pooled = pool_regions(k, v, rid, R)
    truth = exact_region_log_mass(q, k, rid, R, SCALE)
    assert (pooled_logits(q, pooled, SCALE) <= truth + 1e-9).all()


def test_full_rank_covariance_improves_mass_estimate():
    q, k, v, rid, R = _setup(h=8, w=8, g=4, spread=0.3, nq=16)
    truth = exact_region_log_mass(q, k, rid, R, SCALE)
    plain = demand_scores(q, pool_regions(k, v, rid, R, cov_rank=0), SCALE)
    corrected = demand_scores(q, pool_regions(k, v, rid, R, cov_rank=D), SCALE)
    assert (corrected - truth).abs().mean() < (plain - truth).abs().mean()


def test_exact_region_log_mass_matches_bruteforce():
    q, k, v, rid, R = _setup(spread=0.5)
    got = exact_region_log_mass(q, k, rid, R, SCALE)
    logits = SCALE * q @ k.transpose(-1, -2)
    for r in range(R):
        ref = torch.logsumexp(logits[..., rid == r], dim=-1)
        torch.testing.assert_close(got[..., r], ref)
