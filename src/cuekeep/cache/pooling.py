"""Pooled stand-in math for region-level KV paging.

Everything here is pure PyTorch and device-agnostic. It is the *reference* implementation that the
cache layer and the fidelity experiment (E001) both use. See research/method.md for the derivation.

Shape conventions
-----------------
keys, values : [..., n, d]      ... = any leading dims (e.g. batch, kv_heads)
queries      : [..., nq, d]
region_ids   : [n] (long), region index of each visual token, in 0..R-1
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


def region_ids_for_grid(h: int, w: int, g: int, device=None) -> Tensor:
    """Region index for each token of a row-major h×w token grid, tiled by g×g regions.

    Edge regions are smaller when g does not divide h or w. Their size is carried by
    log m, so this needs no special handling.
    """
    if g <= 0:
        raise ValueError("region size g must be positive")
    rows = torch.arange(h, device=device) // g
    cols = torch.arange(w, device=device) // g
    n_col_regions = math.ceil(w / g)
    return (rows[:, None] * n_col_regions + cols[None, :]).reshape(-1)


@dataclass
class PooledRegions:
    keys: Tensor                         # [..., R, d] mean key per region
    values: Tensor                       # [..., R, d] mean value per region
    counts: Tensor                       # [R] tokens per region (m)
    log_m: Tensor                        # [R] log m, the logit bias of each stand-in
    cov_basis: Optional[Tensor] = None   # [..., R, d, r]; Σ_R ≈ B Bᵀ, so qᵀΣq ≈ ‖Bᵀq‖²

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
    """Mean-pool K/V per region. Optionally store a rank-r factor of each region's key covariance."""
    region_ids = region_ids.to(keys.device)
    R = int(num_regions if num_regions is not None else int(region_ids.max()) + 1)
    counts = torch.bincount(region_ids, minlength=R)
    if (counts == 0).any():
        raise ValueError("every region must contain at least one token")
    denom = counts.to(keys.dtype)[:, None]
    k_bar = _scatter_sum(keys, region_ids, R) / denom
    v_bar = _scatter_sum(values, region_ids, R) / denom
    log_m = counts.to(torch.float32).log().to(keys.device)

    cov_basis = None
    if cov_rank > 0:
        cov_basis = _low_rank_covariance(keys, k_bar, region_ids, counts, cov_rank)
    return PooledRegions(k_bar, v_bar, counts, log_m, cov_basis)


def _low_rank_covariance(keys, k_bar, region_ids, counts, rank) -> Tensor:
    """B_R with Σ_R ≈ B_R B_Rᵀ, from the top-r right singular vectors of the centred keys."""
    lead = keys.shape[:-2]
    d = keys.shape[-1]
    R = counts.shape[0]
    flat = keys.reshape(-1, keys.shape[-2], d).float()          # [L, n, d]
    kb = k_bar.reshape(-1, R, d).float()                        # [L, R, d]
    out = flat.new_zeros(flat.shape[0], R, d, rank)
    for r in range(R):
        idx = (region_ids == r).nonzero(as_tuple=True)[0]
        x = (flat[:, idx, :] - kb[:, r : r + 1, :]) / math.sqrt(idx.numel())   # [L, m, d]
        _, s, vh = torch.linalg.svd(x, full_matrices=False)     # vh: [L, k, d], k = min(m, d)
        k = min(rank, s.shape[-1])
        out[:, r, :, :k] = vh[:, :k, :].transpose(-1, -2) * s[:, None, :k]
    return out.reshape(*lead, R, d, rank).to(keys.dtype)


# ----------------------------------------------------------------------------- scores

def pooled_logits(q: Tensor, pooled: PooledRegions, scale: float, use_log_m: bool = True) -> Tensor:
    """Logit of each stand-in: scale·q·k̄ (+ log m). Returns [..., nq, R]."""
    logits = scale * (q @ pooled.keys.transpose(-1, -2))
    if use_log_m:
        logits = logits + pooled.log_m.to(logits.dtype)
    return logits


def demand_scores(q: Tensor, pooled: PooledRegions, scale: float) -> Tensor:
    """Second-order estimate of each region's log attention mass, log Σ_i exp(scale·q·k_i).

    s_R = scale·q·k̄ + log m + ½·scale²·qᵀΣ_R q, with the low-rank Σ_R. Returns [..., nq, R].
    Without a covariance factor this reduces to the pooled logit.
    """
    s = pooled_logits(q, pooled, scale)
    if pooled.cov_basis is not None:
        proj = torch.einsum("...qd,...rdk->...qrk", q, pooled.cov_basis)
        s = s + 0.5 * scale**2 * proj.pow(2).sum(-1)
    return s


def exact_region_log_mass(q: Tensor, keys: Tensor, region_ids: Tensor, num_regions: int, scale: float) -> Tensor:
    """Ground truth log Σ_{i∈R} exp(scale·q·k_i). Returns [..., nq, R]."""
    logits = scale * (q @ keys.transpose(-1, -2)).float()                # [..., nq, n]
    idx = region_ids.to(logits.device).expand_as(logits)
    mx = logits.new_full((*logits.shape[:-1], num_regions), float("-inf"))
    mx = mx.scatter_reduce(-1, idx, logits, reduce="amax", include_self=True)
    shifted = (logits - mx.gather(-1, idx)).exp()
    sums = logits.new_zeros(mx.shape).scatter_add_(-1, idx, shifted)
    return (sums.log() + mx).to(q.dtype)


# ----------------------------------------------------------------------------- attention

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
) -> Tensor:
    """Reference attention over [dense tokens | stand-ins of non-resident regions | exact resident tokens].

    resident: bool [R]. The stand-in of a resident region is masked out, so no region is counted twice.
    Returns [..., nq, d_v].
    """
    region_ids = region_ids.to(q.device)
    resident = resident.to(q.device)
    parts_logit, parts_v = [], []
    if dense_k is not None:
        parts_logit.append(scale * (q @ dense_k.transpose(-1, -2)))
        parts_v.append(dense_v)

    pl = pooled_logits(q, pooled, scale, use_log_m)
    pl = pl.masked_fill(resident, float("-inf"))
    parts_logit.append(pl)
    parts_v.append(pooled.values)

    tok_resident = resident[region_ids]
    if tok_resident.any():
        k_res = region_k[..., tok_resident, :]
        parts_logit.append(scale * (q @ k_res.transpose(-1, -2)))
        parts_v.append(region_v[..., tok_resident, :])

    logits = torch.cat(parts_logit, dim=-1)
    vals = torch.cat(parts_v, dim=-2)
    return torch.softmax(logits.float(), dim=-1).to(vals.dtype) @ vals


def attend_full(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    return torch.softmax(scale * (q @ k.transpose(-1, -2)).float(), dim=-1).to(v.dtype) @ v
