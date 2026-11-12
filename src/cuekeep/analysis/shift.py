"""Does the set of image regions a trace needs shift during decoding? (Hypothesis H2, open question Q3.)

Works on per-step region attention mass, `mass[S, R]`: at each sampled decode step, the attention
each region receives, summed over heads (each head's shares sum to at most 1 with its text share).

The question for paging is whether a set chosen once can serve the whole trace:
  static   top-C regions by mass over the first `warmup` steps, fixed afterwards (the best case for
           prune-once methods, which decide at or just after prefill)
  lagged   top-C regions of the previous sampled step, used at the current step (causal paging at the
           sampling stride)
  dynamic  top-C regions of the current step (oracle paging)
`coverage` is the fraction of visual attention mass that falls on the chosen set. If static coverage
is about equal to dynamic coverage, recall has nothing to add over pruning once.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def per_step_mass(log_shares: Tensor, steps: int) -> Tensor:
    """log_shares [kvh, grp·S, R] (grouped; step of query j is j % S) → mass [S, R], summed over heads."""
    kvh, Q, R = log_shares.shape
    return log_shares.exp().reshape(kvh, Q // steps, steps, R).sum(dim=(0, 1))


def top_c(mass: Tensor, c: int) -> Tensor:
    """Bool [..., R] marking the c highest-mass regions along the last dim."""
    idx = torch.topk(mass, min(c, mass.shape[-1]), dim=-1).indices
    return torch.zeros_like(mass, dtype=torch.bool).scatter_(-1, idx, True)


def coverage(mass: Tensor, chosen: Tensor) -> Tensor:
    """[S]: fraction of each step's visual mass on the chosen regions (chosen: bool [R] or [S, R])."""
    return (mass * chosen).sum(-1) / mass.sum(-1).clamp_min(1e-12)


def shift_metrics(mass: Tensor, frac: float, warmup_frac: float = 0.05) -> Dict[str, float]:
    """Static vs lagged vs dynamic coverage at a budget of `frac` of the regions, after the warmup steps."""
    S, R = mass.shape
    c = max(1, round(frac * R))
    w = max(1, round(warmup_frac * S))
    if S - w < 2:
        raise ValueError("need at least 2 steps after warmup")
    static = top_c(mass[:w].sum(0), c)                   # [R]
    dyn = top_c(mass, c)                                 # [S, R]
    later = slice(w, S)
    cov_dyn = coverage(mass, dyn)[later]
    cov_static = coverage(mass, static)[later]
    cov_lag = coverage(mass[1:], dyn[:-1])[w - 1:]       # set from step s-1, used at step s
    overlap = (dyn[1:] & dyn[:-1]).sum(-1).float() / c
    third = max(1, (S - w) // 3)
    late_sets = dyn[S - third:]
    return {
        "regions": R, "budget_regions": c,
        "cov_dynamic": float(cov_dyn.mean()),
        "cov_static": float(cov_static.mean()),
        "cov_lagged": float(cov_lag.mean()),
        "gap_dynamic_static": float((cov_dyn - cov_static).mean()),
        "gap_dynamic_lagged": float(cov_dyn.mean() - cov_lag.mean()),
        "turnover": float(1 - overlap.mean()),                               # set change between sampled steps
        "late_new_frac": float((late_sets & ~static).sum(-1).float().mean() / c),   # late needs never in the static set
        "visual_mass_early": float(mass[w:w + third].sum(-1).mean()),
        "visual_mass_late": float(mass[S - third:].sum(-1).mean()),
    }
