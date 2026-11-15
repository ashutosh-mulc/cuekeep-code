"""Pooled stand-in math for region-level KV paging.

Everything here is pure PyTorch and device-agnostic. It is the *reference* implementation that the
cache layer and the fidelity experiment (E001) both use. See research/method.md for the derivation.

Precision: pooling, covariance, scores and reference attention are computed in at least float32
(`_acc`): bf16 / fp16 inputs are upcast to float32, and float64 inputs stay float64, so the
exactness tests can check exactness at float64 precision. The pooled K/V are cast back to the input
dtype, because attention consumes them.

Shape conventions
-----------------
keys, values : [..., n, d]      ... = any leading dims (e.g. batch, kv_heads)
queries      : [..., nq, d]
region_ids   : [n] (long), region index of each visual token, in 0..R-1
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

COV_MODES = ("none", "quadratic", "capped")


def _acc(x: Tensor) -> Tensor:
    """x in its accumulation dtype: at least float32 (bf16/fp16 → float32; float64 stays float64)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def region_ids_for_grid(h: int, w: int, g: int, device=None) -> Tensor:
    """Region index for each token of a row-major h×w token grid, tiled by g×g regions.

    Edge regions are smaller when g does not divide h or w. Their size is carried by
    log m, so this needs no special handling.

    Token order: Qwen2/2.5/3-VL processors emit merged tokens row-major over the merged grid (the 2×2
    merge window is folded into each token). Verify once against the installed processor (E000).
    """
    if g <= 0:
        raise ValueError("region size g must be positive")
    rows = torch.arange(h, device=device) // g
    cols = torch.arange(w, device=device) // g
    n_col_regions = math.ceil(w / g)
    return (rows[:, None] * n_col_regions + cols[None, :]).reshape(-1)


def region_ids_for_spans(grids: Sequence[tuple], g: int, device=None) -> Tensor:
    """Region ids for several images, concatenated in sequence order, with ids offset per image.

    grids: [(h, w), ...], one per image (merged-token grid). The caller concatenates the images'
    visual K/V in the same order and ingests them once per layer.
    """
    out, offset = [], 0
    for h, w in grids:
        rid = region_ids_for_grid(h, w, g, device) + offset
        offset = int(rid.max()) + 1
        out.append(rid)
    return torch.cat(out)


@dataclass
class PooledRegions:
    keys: Tensor                         # [..., R, d] mean key per region (input dtype)
    values: Tensor                       # [..., R, d] mean value per region (input dtype)
    counts: Tensor                       # [R] tokens per region (m)
    log_m: Tensor                        # [R] log m (accumulation dtype): the logit bias of each stand-in
    cov_basis: Optional[Tensor] = None   # [..., R, d, r] (accumulation dtype); Σ_R ≈ B Bᵀ, so qᵀΣq ≈ ‖Bᵀq‖²

    @property
    def num_regions(self) -> int:
        return self.counts.shape[0]


def _scatter_sum(x: Tensor, region_ids: Tensor, num_regions: int) -> Tensor:
    dim = x.dim() - 2
    out = x.new_zeros(*x.shape[:-2], num_regions, x.shape[-1])
    return out.index_add_(dim, region_ids, x)


def pool_regions(
    keys: Tensor,
    values: Tensor,
    region_ids: Tensor,
    num_regions: Optional[int] = None,
    cov_rank: int = 0,
) -> PooledRegions:
    """Mean-pool K/V per region (accumulated in ≥ float32). Optionally store a rank-r covariance factor."""
    region_ids = region_ids.to(keys.device)
    R = int(num_regions if num_regions is not None else int(region_ids.max()) + 1)
    counts = torch.bincount(region_ids, minlength=R)
    if (counts == 0).any():
        raise ValueError("every region must contain at least one token")
    k_acc, v_acc = _acc(keys), _acc(values)
    denom = counts.to(k_acc.dtype)[:, None]
    k_bar = _scatter_sum(k_acc, region_ids, R) / denom
    v_bar = _scatter_sum(v_acc, region_ids, R) / denom.to(v_acc.dtype)
    log_m = counts.to(k_acc.dtype).log()

    cov_basis = None
    if cov_rank > 0:
        cov_basis = _low_rank_covariance(k_acc, k_bar, region_ids, counts, cov_rank)
    return PooledRegions(k_bar.to(keys.dtype), v_bar.to(values.dtype), counts, log_m, cov_basis)


def _low_rank_covariance(keys32: Tensor, k_bar32: Tensor, region_ids: Tensor, counts: Tensor, rank: int) -> Tensor:
    """B_R [.., R, d, r] with Σ_R ≈ B_R B_Rᵀ (population covariance), batched over regions of equal size.

    With X_R the centred keys / √m ([m, d]), Σ_R = X_Rᵀ X_R. Rather than an SVD of X_R, use the tiny
    m×m Gram matrix X_R X_Rᵀ = U Λ Uᵀ (m = g² ≤ 64). Then the top-r factor is B_R = X_Rᵀ U_r,
    since B Bᵀ = Xᵀ U_r U_rᵀ X is the rank-r truncation of Σ_R. Regions are grouped by size
    (interior regions are all g², edge regions are smaller), so this is one batched eigh per group.
    """
    lead = keys32.shape[:-2]
    d = keys32.shape[-1]
    R = counts.shape[0]
    flat = keys32.reshape(-1, keys32.shape[-2], d)               # [L, n, d]
    kb = k_bar32.reshape(-1, R, d)                               # [L, R, d]
    out = flat.new_zeros(flat.shape[0], R, d, rank)
    order = torch.argsort(region_ids, stable=True)               # tokens grouped by region
    starts = torch.cumsum(counts, 0) - counts
    for m in torch.unique(counts).tolist():
        regs = (counts == m).nonzero(as_tuple=True)[0]                          # [G]
        tok = order[starts[regs][:, None] + torch.arange(m, device=counts.device)]   # [G, m]
        x = (flat[:, tok, :] - kb[:, regs, None, :]) / math.sqrt(m)             # [L, G, m, d]
        gram = x @ x.transpose(-1, -2)                                          # [L, G, m, m]
        _, u = torch.linalg.eigh(gram)                                          # ascending eigenvalues
        k = min(rank, m)
        u_top = u[..., -k:].flip(-1)                                            # [L, G, m, k], largest first
        out[:, regs, :, :k] = x.transpose(-1, -2) @ u_top                       # [L, G, d, k]
    return out.reshape(*lead, R, d, rank)


# ----------------------------------------------------------------------------- scores

def pooled_logits(q: Tensor, pooled: PooledRegions, scale: float, use_log_m: bool = True) -> Tensor:
    """Logit of each stand-in, scale·q·k̄ (+ log m), in ≥ float32. Returns [..., nq, R]."""
    logits = scale * (_acc(q) @ _acc(pooled.keys).transpose(-1, -2))
    if use_log_m:
        logits = logits + pooled.log_m.to(logits.device, logits.dtype)
    return logits


def demand_scores(q: Tensor, pooled: PooledRegions, scale: float, cov_mode: str = "capped") -> Tensor:
    """Estimate of each region's log attention mass, log Σ_{i∈R} exp(scale·q·k_i), in ≥ float32. [..., nq, R].

    With t = scale·√(qᵀΣ_R q) (low-rank Σ_R), and x_i = scale·q·(k_i − k̄) (mean 0, mean square t²):
      cov_mode="none"       s = scale·q·k̄ + log m                        (Jensen lower bound)
      cov_mode="quadratic"  s = … + ½ t²                                  (second-order / log-normal)
      cov_mode="capped"     s = … + min(½ t², √(m−1)·t)
    The cap is valid because a zero-mean set of m numbers with mean square t² has max ≤ √(m−1)·t, and
    log-mean-exp ≤ max. The quadratic term alone grows without bound and overshoots once logits are
    large (review 2026-11-12). Without a covariance factor, all modes reduce to "none".
    """
    if cov_mode not in COV_MODES:
        raise ValueError(f"cov_mode must be one of {COV_MODES}")
    s = pooled_logits(q, pooled, scale)
    if pooled.cov_basis is None or cov_mode == "none":
        return s
    proj = torch.einsum("...qd,...rdk->...qrk", _acc(q).to(s.dtype), pooled.cov_basis.to(s.dtype))
    t2 = scale**2 * proj.pow(2).sum(-1)                                   # scale² qᵀΣq
    corr = 0.5 * t2
    if cov_mode == "capped":
        cap = (pooled.counts.to(t2) - 1).clamp_min(0).sqrt() * t2.sqrt()
        corr = torch.minimum(corr, cap)
    return s + corr


def region_log_shares(scores: Tensor, other_lse: Optional[Tensor] = None) -> Tensor:
    """Turn per-head region log-mass estimates into per-head log attention shares.

    scores    : [..., nq, R] from demand_scores
    other_lse : [..., nq] log-sum-exp of the logits of every non-visual key the query attends to
                (text and generated tokens). If None, shares are normalised over regions only.
    Shares are comparable across heads and queries, unlike raw log-masses: a head whose logits are
    large but that attends mostly to text must not dominate region selection.
    """
    scores = _acc(scores)
    parts = [scores]
    if other_lse is not None:
        parts.insert(0, other_lse.to(scores.device, scores.dtype).unsqueeze(-1))
    lse = torch.logsumexp(torch.cat(parts, dim=-1), dim=-1, keepdim=True)
    return scores - lse


def exact_region_log_mass(q: Tensor, keys: Tensor, region_ids: Tensor, num_regions: int, scale: float) -> Tensor:
    """Ground truth log Σ_{i∈R} exp(scale·q·k_i), in ≥ float32. Returns [..., nq, R]."""
    logits = scale * (_acc(q) @ _acc(keys).transpose(-1, -2))            # [..., nq, n]
    idx = region_ids.to(logits.device).expand_as(logits)
    mx = logits.new_full((*logits.shape[:-1], num_regions), float("-inf"))
    mx = mx.scatter_reduce(-1, idx, logits, reduce="amax", include_self=True)
    shifted = (logits - mx.gather(-1, idx)).exp()
    sums = logits.new_zeros(mx.shape).scatter_add_(-1, idx, shifted)
    return sums.log() + mx


# ----------------------------------------------------------------------------- attention

def summarize_other(q: Tensor, other_k: Tensor, other_v: Tensor, scale: float):
    """Summary of the non-visual keys for each query: (lse [..., nq], out [..., nq, d]).

    lse is the log-sum-exp of the query's logits over those keys, and out is their attention-weighted
    mean value. Together they reproduce the keys' exact contribution to any attention that also
    includes other entries, so KV dumps store these two small tensors per step instead of the
    whole text KV.
    """
    logits = scale * (_acc(q) @ _acc(other_k).transpose(-1, -2))
    return torch.logsumexp(logits, dim=-1), torch.softmax(logits, dim=-1) @ _acc(other_v).to(logits.dtype)


def _combine(logits: Tensor, values: Tensor, other_lse: Optional[Tensor], other_out: Optional[Tensor]) -> Tensor:
    """Attention output over `logits`/`values`, plus an optional summarised group of other keys."""
    if other_lse is None:
        return torch.softmax(logits, dim=-1) @ values
    lse = other_lse.to(logits.device, logits.dtype).unsqueeze(-1)
    total = torch.logsumexp(torch.cat([lse, logits], dim=-1), dim=-1, keepdim=True)
    return (lse - total).exp() * other_out.to(logits.device, logits.dtype) + (logits - total).exp() @ values


def attend_mixed(
    q: Tensor,
    dense_k: Optional[Tensor],
    dense_v: Optional[Tensor],
    region_k: Tensor,
    region_v: Tensor,
    region_ids: Tensor,
    pooled: PooledRegions,
    resident: Tensor,
    scale: float,
    use_log_m: bool = True,
    standins: bool = True,
    other_lse: Optional[Tensor] = None,
    other_out: Optional[Tensor] = None,
) -> Tensor:
    """Reference attention over [dense tokens | stand-ins of non-resident regions | exact resident tokens].

    resident : bool [R] (one set for every query) or [..., nq, R] (a set per query, e.g. per decode step).
               The stand-in of a resident region is masked out, so no region is counted twice.
    standins : if False, non-resident regions are dropped instead of pooled (prune-style baseline).
    other_lse, other_out : optional summary of further keys (see `summarize_other`), used in place of,
               or in addition to, dense_k / dense_v.
    Logits and softmax are computed in ≥ float32 (`_acc`). Returns [..., nq, d_v] in the value dtype.
    """
    region_ids = region_ids.to(q.device)
    resident = resident.to(q.device)
    qf = _acc(q)
    dt = qf.dtype
    parts_logit, parts_v = [], []
    if dense_k is not None:
        parts_logit.append(scale * (qf @ dense_k.to(dt).transpose(-1, -2)))
        parts_v.append(dense_v.to(dt))

    pl = pooled_logits(q, pooled, scale, use_log_m).to(dt)
    pl = pl.masked_fill(resident, float("-inf")) if standins else torch.full_like(pl, float("-inf"))
    parts_logit.append(pl)
    parts_v.append(pooled.values.to(dt))

    tok_resident = resident[..., region_ids]
    if tok_resident.any():
        if tok_resident.dim() == 1:                       # shared set: attend only over the resident tokens
            parts_logit.append(scale * (qf @ region_k[..., tok_resident, :].to(dt).transpose(-1, -2)))
            parts_v.append(region_v[..., tok_resident, :].to(dt))
        else:                                             # per-query sets: mask the non-resident tokens
            tl = scale * (qf @ region_k.to(dt).transpose(-1, -2))
            parts_logit.append(tl.masked_fill(~tok_resident, float("-inf")))
            parts_v.append(region_v.to(dt))

    out = _combine(torch.cat(parts_logit, dim=-1), torch.cat(parts_v, dim=-2), other_lse, other_out)
    return out.to(region_v.dtype)


def attend_full(q: Tensor, k: Tensor, v: Tensor, scale: float,
                other_lse: Optional[Tensor] = None, other_out: Optional[Tensor] = None) -> Tensor:
    logits = scale * (_acc(q) @ _acc(k).to(_acc(q).dtype).transpose(-1, -2))
    return _combine(logits, v.to(logits.dtype), other_lse, other_out).to(v.dtype)
