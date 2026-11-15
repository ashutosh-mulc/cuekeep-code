"""Correctness of the pooled stand-in math (research/method.md). These run on CPU."""
import math

import pytest

torch = pytest.importorskip("torch")

from cuekeep.cache import (  # noqa: E402
    attend_full, attend_mixed, demand_scores, exact_region_log_mass, pool_regions, pooled_logits,
    region_ids_for_grid, region_ids_for_spans, region_log_shares,
)

D = 16
SCALE = D ** -0.5


def _setup(h=6, w=6, g=2, kvh=2, nq=3, spread=0.0, seed=0, qscale=1.0):
    gen = torch.Generator().manual_seed(seed)
    rid = region_ids_for_grid(h, w, g)
    R = int(rid.max()) + 1
    centres = torch.randn(kvh, R, D, generator=gen, dtype=torch.float64)
    k = centres[:, rid, :] + spread * torch.randn(kvh, h * w, D, generator=gen, dtype=torch.float64)
    v = torch.randn(kvh, h * w, D, generator=gen, dtype=torch.float64)
    q = qscale * torch.randn(kvh, nq, D, generator=gen, dtype=torch.float64)
    return q, k, v, rid, R


# ----------------------------------------------------------------------------- regions

def test_region_ids_tile_grid_with_ragged_edges():
    rid = region_ids_for_grid(5, 7, 2)
    assert rid.shape == (35,)
    counts = torch.bincount(rid)
    assert counts.numel() == math.ceil(5 / 2) * math.ceil(7 / 2)
    assert counts.max() == 4 and counts.min() == 1      # corner region is 1×1
    assert int(counts.sum()) == 35


def test_region_ids_for_spans_offsets_each_image():
    rid = region_ids_for_spans([(4, 4), (2, 6)], g=2)
    assert rid.shape == (16 + 12,)
    assert rid[:16].max() == 3 and rid[16:].min() == 4 and rid[16:].max() == 6
    assert torch.equal(torch.bincount(rid), torch.full((7,), 4))


# ----------------------------------------------------------------------------- stand-ins

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


def test_bf16_pooling_accumulates_in_float32():
    q, k, v, rid, R = _setup(h=16, w=16, g=8, spread=1.0)     # 64 tokens per region
    ref = pool_regions(k, v, rid, R)
    lo = pool_regions(k.bfloat16(), v.bfloat16(), rid, R)
    assert lo.keys.dtype == torch.bfloat16 and lo.log_m.dtype == torch.float32
    torch.testing.assert_close(lo.keys.double(), ref.keys, atol=2e-2, rtol=1e-2)
    s_lo = demand_scores(q.bfloat16(), lo, SCALE)
    assert s_lo.dtype == torch.float32                          # scores are always float32


# ----------------------------------------------------------------------------- scores

def test_jensen_pooled_logit_lower_bounds_true_log_mass():
    q, k, v, rid, R = _setup(spread=0.7)
    pooled = pool_regions(k, v, rid, R)
    truth = exact_region_log_mass(q, k, rid, R, SCALE)
    assert (pooled_logits(q, pooled, SCALE) <= truth + 1e-9).all()


def test_full_rank_covariance_improves_mass_estimate_at_small_logits():
    q, k, v, rid, R = _setup(h=8, w=8, g=4, spread=0.3, nq=16)
    truth = exact_region_log_mass(q, k, rid, R, SCALE)
    plain = demand_scores(q, pool_regions(k, v, rid, R, cov_rank=0), SCALE)
    for mode in ("quadratic", "capped"):
        corrected = demand_scores(q, pool_regions(k, v, rid, R, cov_rank=D), SCALE, cov_mode=mode)
        assert (corrected - truth).abs().mean() < (plain - truth).abs().mean()


def test_covariance_factor_reconstructs_population_covariance():
    q, k, v, rid, R = _setup(h=5, w=7, g=3, spread=0.8)      # ragged regions of sizes 9, 6, 3, 2, 1
    pooled = pool_regions(k, v, rid, R, cov_rank=D)
    for r in range(R):
        x = k[:, rid == r, :]
        xc = x - x.mean(1, keepdim=True)
        sigma = xc.transpose(-1, -2) @ xc / x.shape[1]
        b = pooled.cov_basis[:, r]
        torch.testing.assert_close(b @ b.transpose(-1, -2), sigma, atol=1e-10, rtol=1e-8)   # float64 in, float64 math


def test_capped_correction_never_exceeds_hard_bound():
    q, k, v, rid, R = _setup(h=8, w=8, g=4, spread=1.0, nq=32, qscale=8.0)
    p = pool_regions(k, v, rid, R, cov_rank=D)
    plain = pooled_logits(q, p, SCALE)
    capped = demand_scores(q, p, SCALE, cov_mode="capped")
    quad = demand_scores(q, p, SCALE, cov_mode="quadratic")
    t = ((quad - plain) * 2).sqrt()
    bound = plain + (p.counts.float() - 1).sqrt() * t        # ≥ log m + max_i scale·q·k_i ≥ true log-mass
    assert (capped <= bound + 1e-5).all()
    truth = exact_region_log_mass(q, k, rid, R, SCALE)
    assert (truth <= bound + 1e-5).all()
    assert (capped - truth).abs().mean() < (quad - truth).abs().mean()   # the cap removes overshoot


def test_region_log_shares_are_normalised_per_head():
    gen = torch.Generator().manual_seed(0)
    s = torch.randn(4, 3, 10, generator=gen) * torch.tensor([1.0, 1, 30, 30]).view(4, 1, 1)
    other = torch.randn(4, 3, generator=gen)
    shares = region_log_shares(s, other)
    total = torch.logsumexp(torch.cat([other.unsqueeze(-1), s], -1), -1, keepdim=True)
    torch.testing.assert_close(shares, s - total)
    assert (shares <= 0).all()
    assert torch.allclose(shares.exp().sum(-1) + (other - total.squeeze(-1)).exp(), torch.ones(4, 3))


def test_exact_region_log_mass_matches_bruteforce():
    q, k, v, rid, R = _setup(spread=0.5)
    got = exact_region_log_mass(q, k, rid, R, SCALE)
    logits = SCALE * q @ k.transpose(-1, -2)
    for r in range(R):
        ref = torch.logsumexp(logits[..., rid == r], dim=-1)
        torch.testing.assert_close(got[..., r], ref)
