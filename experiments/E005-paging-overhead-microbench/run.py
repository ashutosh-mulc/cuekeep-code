"""E005: what does paging cost per decode step, next to the decode step itself? (Gate G5, Q4)

No dataset. Random K/V with the real model's shapes, one PagedVisualCache per paged layer, random
queries each step (demand jumps every step: worst-case churn). Per decode step, over all paged layers:
  commit + attention_inputs for every layer, then scoring/planning/fetching
    mode "per_layer": `cache.step(layer, ...)` for each layer on refresh steps (round 0's scoring path)
    mode "batched":   `queue_query` for each layer, then one `end_step` (G5 round 1)
Reported per setting (visual tokens, g, refresh_every, mode):
  total_ms         mean per-step time (one device sync per step), as in round 0
  total_ms_median  median per-step time (robust to a shared GPU)
  score_ms / commit_ms / inputs_ms   the phases timed separately (a sync after each, so they sum to more)
  h2d_mb_per_step, visual_kv_gpu_mib, full_visual_kv_mib

Baselines for the gate:
  pre-registered  `bench.baseline_decode_ms_per_token` (E000's mean, measured in another run)
  in-process      (`baseline_in_process.enabled`) the real model is loaded, and full-KV decode steps at
                  the gate's visual-token count are timed in rounds, interleaved with paging steps at
                  the gate point, so both see the same GPU contention; medians are compared.
`gate.judge` (preregistered | in_process) chooses which verdict is `gate/pass`; both are reported.
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Dict, List

import torch

from cuekeep.cache import PagedVisualCache, region_ids_for_grid


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Pager:
    """One cache per paged layer for one setting, plus a step counter, so rounds can resume it."""

    def __init__(self, n_vis: int, g: int, refresh: int, mode: str, b: dict, device: torch.device, gen):
        self.b, self.device, self.mode, self.refresh = b, device, mode, refresh
        self.kvh, self.heads, self.d, self.layers = b["kv_heads"], b["heads"], b["head_dim"], b["paged_layers"]
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        h = int(math.sqrt(n_vis))
        w = n_vis // h
        rid = region_ids_for_grid(h, w, g)
        self.R = int(rid.max()) + 1
        self.n_vis = h * w
        self.cache = PagedVisualCache({
            "slot_capacity": max(1, round(b["budget"] * self.R)), "fetch_budget": b["fetch_budget"],
            "lookahead": b["lookahead"], "min_residency": b["min_residency"], "cov_rank": 0, "score_agg": "max",
            "refresh_every": refresh,
        }, device=device)
        k = torch.randn(self.kvh, self.n_vis, self.d, generator=gen).to(device, self.dtype)
        self.kv_bytes = 2 * k.numel() * k.element_size()
        for layer in range(self.layers):
            self.cache.ingest(layer, k, k, rid)
        self.t = 0

    def _phases(self, q, lse):
        c, t, L = self.cache, self.t, range(self.layers)

        def commit():
            for l in L:
                c.commit(l, t)

        def inputs():
            for l in L:
                c.attention_inputs(l)

        def score():
            if self.mode == "batched":
                for l in L:
                    c.queue_query(l, q[l], lse[l])
                c.end_step(t)
            elif t % self.refresh == 0:
                for l in L:
                    c.step(l, q[l], t, lse[l])

        return commit, inputs, score

    def step(self, timers: Dict[str, float] = None) -> float:
        """One decode step of paging work. Returns its wall time in ms (synced)."""
        q = torch.randn(self.layers, self.heads, 1, self.d, device=self.device, dtype=self.dtype)
        lse = torch.randn(self.layers, self.heads, 1, device=self.device)
        phases = self._phases(q, lse)
        _sync(self.device)
        t0 = time.perf_counter()
        if timers is None:
            for fn in phases:
                fn()
            _sync(self.device)
        else:
            for name, fn in zip(("commit_ms", "inputs_ms", "score_ms"), phases):
                s = time.perf_counter()
                fn()
                _sync(self.device)
                timers[name] += 1000 * (time.perf_counter() - s)
        self.t += 1
        return 1000 * (time.perf_counter() - t0)


def bench_setting(n_vis: int, g: int, refresh: int, mode: str, b: dict, device, gen) -> Dict[str, float]:
    p = Pager(n_vis, g, refresh, mode, b, device, gen)
    for _ in range(b["warmup_steps"]):
        p.step()
    bytes0 = p.cache.bytes_h2d
    times = [p.step() for _ in range(b["steps"])]
    h2d = (p.cache.bytes_h2d - bytes0) / b["steps"]
    timers = {"commit_ms": 0.0, "inputs_ms": 0.0, "score_ms": 0.0}
    for _ in range(b["steps"]):
        p.step(timers)
    out = {"visual_tokens": p.n_vis, "regions": p.R, "g": g, "refresh_every": refresh, "mode": mode,
           "total_ms": statistics.fmean(times), "total_ms_median": statistics.median(times),
           "h2d_mb_per_step": h2d / 2**20, "visual_kv_gpu_mib": p.cache.gpu_bytes() / 2**20,
           "full_visual_kv_mib": p.layers * p.kv_bytes / 2**20}
    out.update({k: v / b["steps"] for k, v in timers.items()})
    return out


# ----------------------------------------------------------------------------- in-process baseline

class _TokenClock:
    """HF streamer: timestamps each generated token (HF calls put(tokens.cpu()), which syncs)."""

    def __init__(self):
        self.times: List[float] = []
        self._puts = 0

    def put(self, value) -> None:
        self._puts += 1
        if self._puts >= 2:                       # the first put is the prompt
            self.times.append(time.perf_counter())

    def end(self) -> None:
        pass


def decode_step_times(adapter, inputs, tokens: int) -> List[float]:
    """Per-token decode times (ms) of one greedy generate of exactly `tokens` new tokens (prefill excluded)."""
    clock = _TokenClock()
    with torch.inference_mode():
        adapter.model.generate(**inputs, streamer=clock, do_sample=False, temperature=None, top_p=None,
                               top_k=None, max_new_tokens=tokens, min_new_tokens=tokens)
    t = clock.times
    return [1000 * (b - a) for a, b in zip(t[:-1], t[1:])]


def in_process_baseline(cfg: dict, b: dict, device, gen, run) -> Dict[str, object]:
    """Interleave full-KV decode rounds with paging rounds at the gate point; return medians."""
    from PIL import Image

    from cuekeep.models import get_adapter

    ip, gc = cfg["baseline_in_process"], cfg["gate"]
    adapter = get_adapter(cfg["model"], str(device)).load()
    side = int(round(math.sqrt(ip["visual_tokens"])))
    image = Image.new("RGB", (32 * side, 32 * side), (128, 128, 128))
    inputs = adapter.build_inputs(image, "Describe this image in detail.")
    (span,) = adapter.visual_spans(inputs)
    pagers = {k: Pager(gc["visual_tokens"], gc["region_size"], k, gc.get("mode", "batched"), b, device, gen)
              for k in b["refresh_every"] if k <= gc["max_refresh_every"]}
    for p in pagers.values():
        for _ in range(b["warmup_steps"]):
            p.step()
    decode_step_times(adapter, inputs, 4)                         # warm up the model
    dec: List[float] = []
    pag: Dict[int, List[float]] = {k: [] for k in pagers}
    for r in range(ip["rounds"]):
        d = decode_step_times(adapter, inputs, ip["tokens_per_round"])
        dec += d
        for k, p in pagers.items():
            pag[k] += [p.step() for _ in range(ip["tokens_per_round"])]
        run.log(event="baseline_round", round=r, decode_ms_median=statistics.median(d),
                **{f"paging_k{k}_ms_median": statistics.median(v[-ip["tokens_per_round"]:]) for k, v in pag.items()})
    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = {"visual_tokens": span.num_tokens, "decode_ms_median": statistics.median(dec),
           "decode_ms_p90": sorted(dec)[int(0.9 * (len(dec) - 1))], "decode_samples": len(dec)}
    for k, v in pag.items():
        out[f"paging_k{k}_ms_median"] = statistics.median(v)
        out[f"paging_k{k}_ratio"] = statistics.median(v) / out["decode_ms_median"]
    return out


# ----------------------------------------------------------------------------- main

def main(cfg: dict, run) -> dict:
    b = cfg["bench"]
    device = torch.device(cfg["device"])
    gen = torch.Generator().manual_seed(cfg.get("seed", 0))
    baseline = b.get("baseline_decode_ms_per_token")
    rows = []
    for mode in b.get("modes", ["batched"]):
        for n_vis in b["visual_tokens"]:
            for g in b["region_size"]:
                for refresh in b["refresh_every"]:
                    row = bench_setting(n_vis, g, refresh, mode, b, device, gen)
                    if baseline:
                        row["overhead_frac"] = row["total_ms"] / baseline
                        row["commit_frac"] = row["commit_ms"] / baseline
                    rows.append(row)
                    run.log(**row)

    summary: Dict[str, object] = {"device": device.type, "baseline_decode_ms_per_token": baseline}
    for r in rows:
        key = f"{r['mode']}/n{r['visual_tokens']}/g{r['g']}/k{r['refresh_every']}"
        summary[f"{key}/total_ms"] = r["total_ms"]
        summary[f"{key}/total_ms_median"] = r["total_ms_median"]
        if baseline:
            summary[f"{key}/overhead_frac"] = r["overhead_frac"]

    ip = None
    if cfg.get("baseline_in_process", {}).get("enabled"):
        ip = in_process_baseline(cfg, b, device, gen, run)
        summary.update({f"in_process/{k}": v for k, v in ip.items()})

    gc = cfg.get("gate")
    if gc:
        mode = gc.get("mode", "batched")
        ref = [r for r in rows if r["mode"] == mode and r["g"] == gc["region_size"]
               and abs(r["visual_tokens"] - gc["visual_tokens"]) <= gc["visual_tokens"] * 0.05
               and r["refresh_every"] <= gc["max_refresh_every"]]
        best = min(ref, key=lambda r: r["total_ms"]) if ref else None
        on_gpu = device.type == "cuda"
        pre_valid = on_gpu and bool(baseline) and best is not None
        pre_pass = bool(pre_valid and best["overhead_frac"] <= gc["max_overhead_frac"]
                        and best["commit_frac"] <= gc["max_commit_frac"])
        summary.update({
            "gate/mode": mode,
            "gate/best_refresh_every": best["refresh_every"] if best else None,
            "gate/preregistered/valid": pre_valid,
            "gate/preregistered/overhead_frac": best.get("overhead_frac") if best else None,
            "gate/preregistered/commit_frac": best.get("commit_frac") if best else None,
            "gate/preregistered/pass": pre_pass,
        })
        ip_pass = None
        if ip is not None and best is not None:
            ks = [k for k in b["refresh_every"] if k <= gc["max_refresh_every"]]
            k_best = min(ks, key=lambda k: ip[f"paging_k{k}_ratio"])
            commit_frac_ip = best["commit_ms"] / ip["decode_ms_median"]
            ip_pass = bool(on_gpu and ip[f"paging_k{k_best}_ratio"] <= gc["max_overhead_frac"]
                           and commit_frac_ip <= gc["max_commit_frac"])
            summary.update({"gate/in_process/best_refresh_every": k_best,
                            "gate/in_process/overhead_frac": ip[f"paging_k{k_best}_ratio"],
                            "gate/in_process/commit_frac": commit_frac_ip,
                            "gate/in_process/valid": on_gpu,
                            "gate/in_process/pass": ip_pass})
        judge = gc.get("judge", "preregistered")
        summary["gate/judge"] = judge
        summary["gate/pass"] = pre_pass if judge == "preregistered" else bool(ip_pass)
        summary["gate/valid"] = pre_valid if judge == "preregistered" else bool(ip is not None and on_gpu)
    return summary
