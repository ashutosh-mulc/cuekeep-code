"""Baselines. All of them run inside the same cache layer, so memory and latency are measured the same way.

A region selector maps (demand scores or other signals, budget) to the set of resident regions. A
token-level method (FastV, PDrop, ...) is a separate path that physically deletes KV after prefill.
"""
from __future__ import annotations

from typing import Callable, Dict

from .random_regions import random_regions

# name -> (status, source). Update when a baseline lands; tracked in research/roadmap.md (M3).
BASELINES: Dict[str, Dict[str, str]] = {
    "full_kv":        {"status": "planned", "source": "stock HF DynamicCache"},
    "random_regions": {"status": "ready",   "source": "ported from TPRL random_pruner"},
    "fastv":          {"status": "planned", "source": "reimplement; physical delete per Reroute"},
    "pdrop":          {"status": "planned", "source": "reimplement; physical delete per Reroute"},
    "quest_regions":  {"status": "planned", "source": "reimplement Quest min/max keys over regions"},
    "shadowkv_style": {"status": "planned", "source": "port ShadowKV landmarks + value offload"},
    "visionpulse":    {"status": "planned", "source": "reimplement per-step top-k (no public code)"},
    "dstp":           {"status": "planned", "source": "reimplement restore-on-shift (no public code)"},
    "snapkv":         {"status": "planned", "source": "kvpress scoring"},
    "vimod":          {"status": "planned", "source": "reimplement or match reported settings (no public code)"},
}

SELECTORS: Dict[str, Callable] = {"random_regions": random_regions}
