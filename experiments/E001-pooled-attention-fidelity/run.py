"""E001: fidelity of the pooled stand-ins and of the demand score.

For each (layer/head unit, g, r) this computes:
  rel_err_logm   ‖attend_mixed(all pooled, with log m) − attend_full‖ / ‖attend_full‖
  rel_err_nobias the same without the log m bias
  spearman       rank correlation between the demand score and the exact region log-mass
  topb_recall    overlap of the top-B regions by demand score with the top-B by exact mass
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch

from cuekeep.cache import (
    attend_full, attend_mixed, demand_scores, exact_region_log_mass, pool_regions, region_ids_for_grid,
)


def _rank(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    num = (ra * rb).sum(-1)
    den = ra.norm(dim=-1) * rb.norm(dim=-1)
    return float((num / den.clamp_min(1e-9)).mean())


def topb_recall(score: torch.Tensor, truth: torch.Tensor, b: int) -> float:
    b = min(b, score.shape[-1])
    s = torch.topk(score, b, dim=-1).indices
    t = torch.topk(truth, b, dim=-1).indices
    hits = (s.unsqueeze(-1) == t.unsqueeze(-2)).any(-1).float().sum(-1)
    return float((hits / b).mean())


# ----------------------------------------------------------------------------- sources

def synthetic_units(cfg: dict, g: int, gen: torch.Generator) -> Iterator[Tuple[str, dict]]:
    sc = cfg["synthetic"]
    h, w = sc["grid"]
    kvh, d, nq = sc["kv_heads"], sc["head_dim"], sc["num_queries"]
    rid = region_ids_for_grid(h, w, g)
    R = int(rid.max()) + 1
    for spread in sc["spreads"]:
        centres = torch.randn(kvh, R, d, generator=gen)
        k = centres[:, rid, :] + spread * torch.randn(kvh, h * w, d, generator=gen)
        v = torch.randn(kvh, h * w, d, generator=gen)
        q = torch.randn(kvh, nq, d, generator=gen)
        nt = sc.get("text_tokens", 0)
        tk = torch.randn(kvh, nt, d, generator=gen) if nt else None
        tv = torch.randn(kvh, nt, d, generator=gen) if nt else None
        yield f"spread={spread}", {"k": k, "v": v, "q": q, "grid": (h, w), "text_k": tk, "text_v": tv}


def dump_units(cfg: dict) -> Iterator[Tuple[str, dict]]:
    """The dump file holds a list of {sample_id, layer, k [kvh,n,d], v, q [kvh,nq,d], grid (h,w),
    text_k/text_v [kvh,nt,d] (the non-visual KV at the same layer), rope: pre|post}."""
    data = torch.load(cfg["dump_path"], map_location="cpu")
    for u in data:
        yield f"{u['sample_id']}/L{u['layer']}/{u.get('rope', 'post')}", u


# ----------------------------------------------------------------------------- main

def evaluate_unit(u: dict, g: int, ranks: List[int], top_b: int) -> Dict[str, float]:
    k, v, q = u["k"].float(), u["v"].float(), u["q"].float()
    h, w = u["grid"]
    rid = region_ids_for_grid(h, w, g)
    R = int(rid.max()) + 1
    scale = k.shape[-1] ** -0.5
    # The log m bias only matters relative to entries of a different size: text tokens, resident
    # exact tokens, ragged edge regions. Without text tokens and with equal regions it cancels in softmax.
    tk, tv = u.get("text_k"), u.get("text_v")
    tk = tk.float() if tk is not None else None
    tv = tv.float() if tv is not None else None
    full = attend_full(q, torch.cat([tk, k], 1), torch.cat([tv, v], 1), scale) if tk is not None \
        else attend_full(q, k, v, scale)
    none_res = torch.zeros(R, dtype=torch.bool)
    out: Dict[str, float] = {}
    base = pool_regions(k, v, rid, R, cov_rank=0)
    for use_bias, key in ((True, "rel_err_logm"), (False, "rel_err_nobias")):
        approx = attend_mixed(q, tk, tv, k, v, rid, base, none_res, scale, use_log_m=use_bias)
        out[key] = float(((approx - full).norm(dim=-1) / full.norm(dim=-1).clamp_min(1e-9)).median())
    truth = exact_region_log_mass(q, k, rid, R, scale)
    for r in ranks:
        pooled = base if r == 0 else pool_regions(k, v, rid, R, cov_rank=r)
        s = demand_scores(q, pooled, scale)
        out[f"spearman_r{r}"] = spearman(s, truth)
        out[f"topb_recall_r{r}"] = topb_recall(s, truth, top_b)
        out[f"mass_mae_r{r}"] = float((s - truth).abs().mean())
    return out


def main(cfg: dict, run) -> dict:
    gen = torch.Generator().manual_seed(cfg.get("seed", 0))
    sw = cfg["sweep"]
    rows = []
    for g in sw["region_size"]:
        units = synthetic_units(cfg, g, gen) if cfg["source"] == "synthetic" else dump_units(cfg)
        for name, u in units:
            m = evaluate_unit(u, g, sw["cov_rank"], sw["top_b"])
            row = {"unit": name, "g": g, **m}
            rows.append(row)
            run.log(**row)

    # Aggregate per (g) and per (g, unit) for the README table.
    summary: Dict[str, float] = {}
    for g in sw["region_size"]:
        sel = [r for r in rows if r["g"] == g]
        for key in sel[0]:
            if key in ("unit", "g"):
                continue
            summary[f"g{g}/{key}"] = statistics.median(r[key] for r in sel)
    out = Path(run.dir) / "rows.json"
    out.write_text(json.dumps(rows, indent=1))
    return summary
