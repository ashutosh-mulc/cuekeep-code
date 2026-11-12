"""E005: what does paging cost per decode step, next to the decode step itself? (Gate G5, Q4)

No dataset and no model: random K/V with the real model's shapes, one PagedVisualCache per paged
layer, and random queries each step. Random queries make region demand jump every step, so fetch
churn is at its worst case.

Per setting (visual tokens, region size, refresh interval) it reports, per decode step summed over
the paged layers:
  total_ms    score + plan + issue fetches + commit + build attention inputs, with one device sync per step
  score_ms / commit_ms / inputs_ms    the same phases timed separately (a sync after each, so they sum to more)
  h2d_mb_per_step, fetches_per_step
The gate compares total_ms with the full-KV decode step time measured in E000
(`bench.baseline_decode_ms_per_token`, passed as an override).
"""
from __future__ import annotations

import math
import time
from typing import Dict

import torch

from cuekeep.cache import PagedVisualCache, region_ids_for_grid


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_setting(n_vis: int, g: int, refresh: int, b: dict, device: torch.device, gen: torch.Generator) -> Dict[str, float]:
    kvh, heads, d, layers = b["kv_heads"], b["heads"], b["head_dim"], b["paged_layers"]
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    h = int(math.sqrt(n_vis))
    w = n_vis // h
    rid = region_ids_for_grid(h, w, g)
    R = int(rid.max()) + 1
    cache = PagedVisualCache({
        "slot_capacity": max(1, round(b["budget"] * R)), "fetch_budget": b["fetch_budget"],
        "lookahead": b["lookahead"], "min_residency": b["min_residency"], "cov_rank": 0, "score_agg": "max",
    }, device=device)
    k = torch.randn(kvh, h * w, d, generator=gen).to(device, dtype)
    for layer in range(layers):
        cache.ingest(layer, k, k, rid)

    def one_step(step: int, timers: Dict[str, float] = None) -> None:
        q = torch.randn(heads, 1, d, device=device, dtype=dtype)
        lse = torch.randn(heads, 1, device=device)
        phases = (
            ("score_ms", lambda l: cache.step(l, q, step, lse) if step % refresh == 0 else None),
            ("commit_ms", lambda l: cache.commit(l, step)),
            ("inputs_ms", lambda l: cache.attention_inputs(l)),
        )
        if timers is None:                               # realistic order: all three phases per layer
            for layer in range(layers):
                for _, fn in phases:
                    fn(layer)
            return
        for name, fn in phases:                          # phase by phase, with a sync after each
            t0 = time.perf_counter()
            for layer in range(layers):
                fn(layer)
            _sync(device)
            timers[name] += 1000 * (time.perf_counter() - t0)

    for s in range(b["warmup_steps"]):
        one_step(s)
    _sync(device)
    bytes0 = cache.bytes_h2d
    t0 = time.perf_counter()
    for s in range(b["warmup_steps"], b["warmup_steps"] + b["steps"]):
        one_step(s)
        _sync(device)
    total_ms = 1000 * (time.perf_counter() - t0) / b["steps"]
    h2d = (cache.bytes_h2d - bytes0) / b["steps"]

    timers = {"score_ms": 0.0, "commit_ms": 0.0, "inputs_ms": 0.0}
    base = b["warmup_steps"] + b["steps"]
    for s in range(base, base + b["steps"]):
        one_step(s, timers)
    out = {"visual_tokens": h * w, "regions": R, "g": g, "refresh_every": refresh, "total_ms": total_ms,
           "h2d_mb_per_step": h2d / 2**20, "fetches_per_step": h2d / max(2 * kvh * g * g * d * k.element_size(), 1),
           "visual_kv_gpu_mib": cache.gpu_bytes() / 2**20,
           "full_visual_kv_mib": layers * 2 * k.numel() * k.element_size() / 2**20}
    out.update({name: ms / b["steps"] for name, ms in timers.items()})
    return out


def main(cfg: dict, run) -> dict:
    b = cfg["bench"]
    device = torch.device(cfg["device"])
    gen = torch.Generator().manual_seed(cfg.get("seed", 0))
    baseline = b.get("baseline_decode_ms_per_token")
    rows = []
    for n_vis in b["visual_tokens"]:
        for g in b["region_size"]:
            for refresh in b["refresh_every"]:
                row = bench_setting(n_vis, g, refresh, b, device, gen)
                if baseline:
                    row["overhead_frac"] = row["total_ms"] / baseline
                    row["commit_frac"] = row["commit_ms"] / baseline
                rows.append(row)
                run.log(**row)

    summary: Dict[str, object] = {"device": device.type, "baseline_decode_ms_per_token": baseline}
    for r in rows:
        key = f"n{r['visual_tokens']}/g{r['g']}/k{r['refresh_every']}"
        summary[f"{key}/total_ms"] = r["total_ms"]
        if baseline:
            summary[f"{key}/overhead_frac"] = r["overhead_frac"]
    gc = cfg.get("gate")
    if gc:
        ref = [r for r in rows if r["g"] == gc["region_size"] and abs(r["visual_tokens"] - gc["visual_tokens"]) <= gc["visual_tokens"] * 0.05
               and r["refresh_every"] <= gc["max_refresh_every"]]
        valid = device.type == "cuda" and bool(baseline) and bool(ref)
        best = min(ref, key=lambda r: r["total_ms"]) if ref else None
        summary.update({
            "gate/valid": valid,                         # needs CUDA and E000's decode ms/token
            "gate/best_refresh_every": best["refresh_every"] if best else None,
            "gate/overhead_frac": best.get("overhead_frac") if best else None,
            "gate/commit_frac": best.get("commit_frac") if best else None,
            "gate/pass": bool(valid and best["overhead_frac"] <= gc["max_overhead_frac"]
                              and best["commit_frac"] <= gc["max_commit_frac"]),
        })
    return summary
