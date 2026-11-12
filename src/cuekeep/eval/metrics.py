"""Systems and accuracy metrics. Every method reports the same set (research/method.md § What we measure).

Memory is reported per phase and per component, because a whole-`generate` peak hides the effect we
claim. Prefill holds the full visual KV for every method, and model weights (about 9 GB for the 4B model)
dwarf a single request's visual KV (about 150 MB for 1k visual tokens). The headline numbers are:
  visual_kv_gpu_bytes   device bytes for visual KV during decode (full KV: all of it; CueKeep: pooled + slots)
  decode_peak_bytes     torch peak allocation during the decode phase only (reset after the first token)
"""
from __future__ import annotations

import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch


@dataclass
class GenerationRecord:
    sample_id: str
    correct: Optional[bool]
    num_new_tokens: int
    seconds: float                              # whole request, prefill (incl. vision encoder) + decode
    prefill_seconds: Optional[float] = None     # time to first generated token
    truncated: bool = False                     # hit max_new_tokens without EOS (trace length is censored)
    prompt_tokens: int = 0
    visual_tokens: int = 0
    kv_bytes_per_token: int = 0                 # all layers, K+V, in the cache dtype
    visual_kv_gpu_bytes: Optional[int] = None   # device bytes for visual KV during decode
    prefill_peak_bytes: Optional[int] = None
    decode_peak_bytes: Optional[int] = None
    h2d_bytes: int = 0
    exact_mass_frac: Optional[float] = None     # fraction of attention mass served by exact KV (paging only)

    @property
    def decode_seconds(self) -> Optional[float]:
        return None if self.prefill_seconds is None else self.seconds - self.prefill_seconds

    @property
    def ms_per_token(self) -> float:
        """Decode-only latency per token. NaN if no prefill split was recorded (never mixes prefill in)."""
        if self.decode_seconds is None:
            return float("nan")
        return 1000.0 * self.decode_seconds / max(self.num_new_tokens - 1, 1)   # token 1 comes from prefill

    @property
    def visual_kv_bytes(self) -> int:
        return self.visual_tokens * self.kv_bytes_per_token

    @property
    def total_kv_bytes_end(self) -> int:
        return (self.prompt_tokens + self.num_new_tokens) * self.kv_bytes_per_token


class PhaseTracker:
    """Splits a `model.generate` call into prefill and decode, for both time and peak memory.

    Pass it as `streamer=`. HF generate calls `put(prompt_ids)` once, then `put(tokens)` after each
    generated token, then `end()`. On the second `put` (the first generated token, which comes from
    the prefill forward) we record the prefill time and peak, then reset the CUDA peak counter, so the
    peak read at `end()` covers decode only. The one synchronize at that point is the only overhead.
    Duck-typed. If the installed transformers insists on a BaseStreamer subclass, wrap this in one.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.cuda = torch.cuda.is_available()
        self.device = device
        self._puts = 0
        self.t0 = self.t_first = self.t_end = None
        self.start_alloc = self.prefill_peak = self.decode_peak = None

    def _sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def start(self) -> "PhaseTracker":
        self._sync()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_alloc = int(torch.cuda.memory_allocated(self.device))
        self.t0 = time.perf_counter()
        return self

    def put(self, value) -> None:
        self._puts += 1
        if self._puts == 2:
            self._sync()
            self.t_first = time.perf_counter()
            if self.cuda:
                self.prefill_peak = int(torch.cuda.max_memory_allocated(self.device))
                torch.cuda.reset_peak_memory_stats(self.device)

    def end(self) -> None:
        self._sync()
        self.t_end = time.perf_counter()
        if self.cuda:
            self.decode_peak = int(torch.cuda.max_memory_allocated(self.device))

    @property
    def seconds(self) -> Optional[float]:
        return None if self.t_end is None else self.t_end - self.t0

    @property
    def prefill_seconds(self) -> Optional[float]:
        return None if self.t_first is None else self.t_first - self.t0


@contextmanager
def cuda_peak_memory(device: Optional[int] = None) -> Iterator[Dict[str, int]]:
    """Whole-block peak. Prefer PhaseTracker for generation; this one includes prefill. 0 on CPU."""
    out: Dict[str, int] = {"peak_bytes": 0}
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            out["peak_bytes"] = int(torch.cuda.max_memory_allocated(device))


@contextmanager
def wall_timer() -> Iterator[Dict[str, float]]:
    out: Dict[str, float] = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out["seconds"] = time.perf_counter() - t0


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


@dataclass
class Summary:
    records: List[GenerationRecord] = field(default_factory=list)

    def add(self, r: GenerationRecord) -> None:
        self.records.append(r)

    def as_dict(self) -> Dict[str, float]:
        rs = self.records
        if not rs:
            return {}
        scored = [r for r in rs if r.correct is not None]
        n_tok = sum(r.num_new_tokens for r in rs)
        mib = 2**20
        out = {
            "n": len(rs),
            "accuracy": (sum(r.correct for r in scored) / len(scored)) if scored else float("nan"),
            "mean_trace_tokens": n_tok / len(rs),
            "median_trace_tokens": statistics.median(r.num_new_tokens for r in rs),
            "truncation_rate": sum(r.truncated for r in rs) / len(rs),     # >0 means trace lengths are censored
            "mean_response_seconds": _mean([r.seconds for r in rs]),
            "mean_prefill_seconds": _mean([r.prefill_seconds for r in rs]),
            "mean_decode_ms_per_token": _mean([r.ms_per_token for r in rs]),
            "h2d_bytes_per_token": sum(r.h2d_bytes for r in rs) / max(n_tok, 1),
            "mean_visual_tokens": _mean([r.visual_tokens for r in rs]),
            "mean_visual_kv_mib": _mean([r.visual_kv_bytes / mib for r in rs]),
            "mean_total_kv_end_mib": _mean([r.total_kv_bytes_end / mib for r in rs]),
        }
        gpu_vis = [r.visual_kv_gpu_bytes / mib for r in rs if r.visual_kv_gpu_bytes is not None]
        if gpu_vis:
            out["mean_visual_kv_gpu_mib"] = _mean(gpu_vis)
        for key, attr in (("max_prefill_peak_gib", "prefill_peak_bytes"), ("max_decode_peak_gib", "decode_peak_bytes")):
            vals = [getattr(r, attr) for r in rs if getattr(r, attr)]
            if vals:
                out[key] = max(vals) / 2**30
        return out
