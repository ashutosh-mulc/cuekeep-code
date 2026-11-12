"""E001: fidelity of the pooled stand-ins, and how well region scores pick what to fetch.

For each unit (one sample × layer: visual K/V, decode queries, and the text keys' summary) and each g:
  capture_rel_err                 (dump only) our recomputed attention vs the model's recorded output
  rel_err_logm / rel_err_nobias   output error with every region pooled, with and without the log m bias
  budget<f>/oracle_standins       output error when, at each decode step, the top f of regions by true
  budget<f>/oracle_drop           attention mass are exact and the rest are pooled / dropped
  mass/<variant>/{spearman,recall,mae}     score variant vs the exact region log-mass (per head, query)
  benefit/<variant>/{spearman,recall}      score variant vs the oracle benefit of fetching one region
  agg/<rule>/{spearman,recall}             cross-head aggregation rules vs the head-averaged oracle benefit
Variants: plain (r=0), and for each r in the sweep, quadratic and capped (see pooling.demand_scores).

Tiers (research/experiment_design.md):
  config.synthetic.yaml  CPU code check on a synthetic field. Its numbers are not results.
  config.gate.yaml       gate G2, on the gate dump. The verdict is computed into summary["gate/..."].
  config.main.yaml       main tier on the main dump; held until G2 is passed.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch

from cuekeep.analysis import iter_dump, per_step_mass, to_unit, top_c
from cuekeep.cache import (
    attend_full, attend_mixed, demand_scores, exact_region_log_mass, pool_regions, pooled_logits,
    region_ids_for_grid, region_log_shares,
)


def _rank(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return float(((ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-9)).mean())


def topb_recall(score: torch.Tensor, truth: torch.Tensor, b: int) -> float:
    b = min(b, score.shape[-1])
    s = torch.topk(score, b, dim=-1).indices
    t = torch.topk(truth, b, dim=-1).indices
    return float(((s.unsqueeze(-1) == t.unsqueeze(-2)).any(-1).float().sum(-1) / b).mean())


# ----------------------------------------------------------------------------- sources

def smooth_field(h: int, w: int, d: int, lead: int, n_waves: int, gen: torch.Generator) -> torch.Tensor:
    """[lead, h*w, d] random low-frequency field: a sum of 2D cosines with random vector amplitudes."""
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    pos = torch.stack([ys.reshape(-1) / h, xs.reshape(-1) / w], -1)                 # [n, 2]
    freq = torch.rand(lead, n_waves, 2, generator=gen) * 3.0                        # up to 3 cycles per image
    phase = torch.rand(lead, n_waves, 1, generator=gen) * 2 * math.pi
    amp = torch.randn(lead, n_waves, d, generator=gen)
    waves = torch.cos(2 * math.pi * torch.einsum("lwk,nk->lwn", freq, pos) + phase)   # [lead, W, n]
    field = torch.einsum("lwn,lwd->lnd", waves, amp)
    return field / field.std(dim=1, keepdim=True).clamp_min(1e-6)                   # unit variance per dim


def synthetic_units(cfg: dict, gen: torch.Generator) -> Iterator[dict]:
    """Raw units in the text-KV form; `to_unit` summarises the text keys, as it does for dumps."""
    sc = cfg["synthetic"]
    h, w = sc["grid"]
    kvh, d, nq, nt = sc["kv_heads"], sc["head_dim"], sc["num_queries"], sc["text_tokens"]
    for noise in sc["noise"]:
        k = smooth_field(h, w, d, kvh, sc["n_waves"], gen) + noise * torch.randn(kvh, h * w, d, generator=gen)
        v = smooth_field(h, w, d, kvh, sc["n_waves"], gen) + noise * torch.randn(kvh, h * w, d, generator=gen)
        tk = torch.randn(kvh, nt, d, generator=gen)
        tv = torch.randn(kvh, nt, d, generator=gen)
        for qs in sc["query_scales"]:
            # Logit std ≈ qs (keys have ~unit variance per dim and scale = 1/√d). Per-head gains make
            # some heads "loud", as in real models.
            head_gain = torch.exp(torch.randn(kvh, 1, 1, generator=gen) * sc.get("head_gain_std", 0.5))
            q = qs * head_gain * torch.randn(kvh, nq, d, generator=gen)
            yield {"name": f"noise={noise}/qscale={qs}", "dataset": "synthetic", "rope": "post",
                   "k": k, "v": v, "q": q, "grid": (h, w), "text_k": tk, "text_v": tv}


def raw_units(cfg: dict, gen: torch.Generator) -> Iterator[dict]:
    if cfg["source"] == "synthetic":
        return synthetic_units(cfg, gen)
    if not cfg.get("dump_path"):
        raise ValueError("source=dump needs dump_path=<directory written by scripts/dump_visual_kv.py>")
    return iter_dump(cfg["dump_path"])


# ----------------------------------------------------------------------------- evaluation

def _rel_err(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).norm(dim=-1) / b.float().norm(dim=-1).clamp_min(1e-9)


def evaluate_unit(u: dict, g: int, sw: dict, attn_out=None) -> Dict[str, float]:
    k, v, q, lse, o_out = u["k"], u["v"], u["q"], u["other_lse"], u["other_out"]
    kvh, nq, d = q.shape
    h, w = u["grid"]
    rid = region_ids_for_grid(h, w, g, device=k.device)
    R = int(rid.max()) + 1
    scale = d ** -0.5
    top_b = sw["top_b"]
    # The log m bias only matters relative to entries of a different size: text tokens, resident
    # exact tokens, ragged edge regions. Without those it cancels in the softmax.
    full = attend_full(q, k, v, scale, lse, o_out)

    out: Dict[str, float] = {"logit_std": float((scale * q @ k.transpose(-1, -2)).std()), "regions": R}
    if attn_out is not None:                      # capture check: is the dumped q what attention really used?
        out["capture_rel_err"] = float(_rel_err(full, attn_out.to(full.device).reshape(kvh, nq, -1)).median())

    none_res = torch.zeros(R, dtype=torch.bool, device=k.device)
    base = pool_regions(k, v, rid, R, cov_rank=0)
    err0 = None
    for use_bias, key in ((True, "rel_err_logm"), (False, "rel_err_nobias")):
        e = _rel_err(attend_mixed(q, None, None, k, v, rid, base, none_res, scale, use_log_m=use_bias,
                                  other_lse=lse, other_out=o_out), full)
        out[key] = float(e.median())
        if use_bias:
            err0 = e                                                                    # [kvh, nq]

    truth = exact_region_log_mass(q, k, rid, R, scale)                                  # [kvh, nq, R]

    # Error at a budget, with oracle residency chosen per decode step from the true attention mass.
    S = u["steps"]
    mass = per_step_mass(region_log_shares(truth, lse), S)                              # [S, R]
    for frac in sw.get("budgets", []):
        res_step = top_c(mass, max(1, round(frac * R)))                                 # [S, R]
        res = res_step.repeat(nq // S, 1).unsqueeze(0).expand(kvh, nq, R)               # step of query j is j % S
        for standins, key in ((True, "oracle_standins"), (False, "oracle_drop")):
            e = _rel_err(attend_mixed(q, None, None, k, v, rid, base, res, scale, standins=standins,
                                      other_lse=lse, other_out=o_out), full)
            out[f"budget{frac}/{key}"] = float(e.median())

    # Score variants.
    variants = {"plain": pooled_logits(q, base, scale)}
    for r in sw["cov_rank"]:
        if r == 0:
            continue
        pr = pool_regions(k, v, rid, R, cov_rank=r)
        for mode in sw["cov_mode"]:
            variants[f"r{r}_{mode}"] = demand_scores(q, pr, scale, cov_mode=mode)
    for name, s in variants.items():
        out[f"mass/{name}/spearman"] = spearman(s, truth)
        out[f"mass/{name}/recall"] = topb_recall(s, truth, top_b)
        out[f"mass/{name}/mae"] = float((s - truth).abs().mean())

    if sw.get("oracle_benefit", True) and R <= sw.get("oracle_max_regions", 256):
        benefit = torch.empty(kvh, nq, R, device=k.device)
        for r in range(R):
            res1 = none_res.clone()
            res1[r] = True
            benefit[..., r] = err0 - _rel_err(attend_mixed(q, None, None, k, v, rid, base, res1, scale,
                                                           other_lse=lse, other_out=o_out), full)
        for name, s in variants.items():
            out[f"benefit/{name}/spearman"] = spearman(s, benefit)
            out[f"benefit/{name}/recall"] = topb_recall(s, benefit, top_b)
        agg_truth = benefit.reshape(-1, R).mean(0, keepdim=True)
        rules = {}
        for name in ("plain", *(n for n in variants if n.endswith("_capped"))):
            s = variants[name]
            shares = region_log_shares(s, lse).reshape(-1, R)
            rules[f"{name}/rawmax"] = s.reshape(-1, R).amax(0, keepdim=True)
            rules[f"{name}/share_max"] = shares.amax(0, keepdim=True)
            rules[f"{name}/share_mean"] = (torch.logsumexp(shares, 0) - math.log(shares.shape[0]))[None]
        for name, s in rules.items():
            out[f"agg/{name}/spearman"] = spearman(s, agg_truth)
            out[f"agg/{name}/recall"] = topb_recall(s, agg_truth, top_b)
    return out


# ----------------------------------------------------------------------------- gate verdict

def _frac(rows: List[dict], pred) -> float:
    return sum(1 for r in rows if pred(r)) / max(len(rows), 1)


def gate_verdict(rows: List[dict], gc: dict) -> Dict[str, object]:
    """G2 pass criteria (research/experiment_design.md), on post-RoPE units. Returns values and booleans."""
    post = [r for r in rows if r.get("rope", "post") == "post"]
    out: Dict[str, object] = {"gate/units": len({r["unit"] for r in post})}
    # 1. Capture check.
    caps = [r["capture_rel_err"] for r in post if "capture_rel_err" in r]
    out["gate/capture_max_rel_err"] = max(caps) if caps else None
    c1 = bool(caps) and max(caps) < gc["capture_max_rel_err"]
    # 2. The log m bias helps, at every g.
    by_g = {g: [r for r in post if r["g"] == g] for g in sorted({r["g"] for r in post})}
    bias = {g: _frac(rs, lambda r: r["rel_err_logm"] < r["rel_err_nobias"]) for g, rs in by_g.items()}
    out["gate/logm_helps_frac_min"] = min(bias.values())
    c2 = min(bias.values()) >= gc["logm_helps_min_frac"]
    # 3. At the gate budget, oracle residency + stand-ins is accurate, and beats dropping the rest.
    f = gc["budget"]
    ok, beats = {}, {}
    for g in gc["region_sizes"]:
        rs = by_g.get(g, [])
        ok[g] = _frac(rs, lambda r: r[f"budget{f}/oracle_standins"] <= gc["max_rel_err"])
        beats[g] = _frac(rs, lambda r: r[f"budget{f}/oracle_standins"] < r[f"budget{f}/oracle_drop"])
    out["gate/budget_ok_frac_min"] = min(ok.values())
    out["gate/standins_beat_drop_frac_min"] = min(beats.values())
    c3 = min(ok.values()) >= gc["min_frac"] and min(beats.values()) >= gc["min_frac"]
    out.update({"gate/c1_capture": c1, "gate/c2_logm_bias": c2, "gate/c3_budget_fidelity": c3,
                "gate/pass": c1 and c2 and c3})
    return out


def main(cfg: dict, run) -> dict:
    gen = torch.Generator().manual_seed(cfg.get("seed", 0))
    sw = cfg["sweep"]
    device = cfg["device"]
    rows = []
    for raw in raw_units(cfg, gen):                    # one unit in memory at a time; every g sees the same unit
        u = to_unit(raw, device)
        for g in sw["region_size"]:
            row = {"unit": u["name"], "dataset": u["dataset"], "layer": u["layer"],
                   "rope": raw.get("rope", "post"), "g": g, **evaluate_unit(u, g, sw, raw.get("attn_out"))}
            rows.append(row)
            run.log(**row)

    summary: Dict[str, object] = {"units": len({r["unit"] for r in rows})}
    for rope in sorted({r["rope"] for r in rows}):
        for g in sw["region_size"]:
            sel = [r for r in rows if r["g"] == g and r["rope"] == rope]
            keys = set.intersection(*(set(r) for r in sel)) - {"unit", "dataset", "layer", "rope", "g"}
            prefix = f"g{g}/" if rope == "post" else f"{rope}/g{g}/"
            for key in sorted(keys):
                summary[prefix + key] = statistics.median(r[key] for r in sel)
    if cfg.get("gate"):
        summary.update(gate_verdict(rows, cfg["gate"]))
    (Path(run.dir) / "rows.json").write_text(json.dumps(rows, indent=1))
    return summary
