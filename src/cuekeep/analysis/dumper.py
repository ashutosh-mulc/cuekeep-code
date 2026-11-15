"""Produce KV dump units (analysis/dumps.py format) for one sample: generate, then capture.

1. Generate the trace with the model's own decoding and a fixed per-sample seed.
2. Choose decode steps: `spread_steps` spread evenly over the trace, and optionally one dense window
   of consecutive steps (for the 1-step-lag check in E003).
3. One teacher-forced pass over prompt + trace with `KVCapture` on the chosen layers.
4. Write one `.pt` file per (layer, kind), and return a manifest row (trace length, truncation,
   prompt and visual token counts, and the ScreenSpot-Pro click score when there is a box).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .capture import KVCapture, pad_like_input_ids


def choose_steps(n: int, spread: int, dense_len: int = 0, dense_start_frac: float = 0.25,
                 dense_min: int = 32) -> Dict[str, torch.Tensor]:
    """Decode-step indices (0-based) per kind. Spread: evenly over [0, n). Dense: consecutive, if long enough."""
    out = {"spread": torch.linspace(0, n - 1, min(spread, n)).round().long().unique()}
    if dense_len > 0:
        start = int(dense_start_frac * n)
        length = min(dense_len, n - start)
        if length >= dense_min:
            out["dense"] = torch.arange(start, start + length)
    return out


def dump_sample(adapter, sample: Dict[str, Any], seed: int, layers: List[int], out_dir: Path,
                spread_steps: int = 64, dense_len: int = 0, dense_start_frac: float = 0.25,
                min_trace: int = 16, dataset: str = "?") -> Dict[str, Any]:
    t0 = time.perf_counter()
    inputs = adapter.build_inputs(sample["image"], sample["question"], sample.get("system"))
    spans = adapter.visual_spans(inputs)
    row: Dict[str, Any] = {"dataset": dataset, "sample_id": sample["id"], "seed": seed,
                           "prompt_tokens": int(inputs["input_ids"].shape[1])}
    if len(spans) != 1:
        return {**row, "status": f"skipped: {len(spans)} images (need exactly 1)"}
    sp = spans[0]
    row.update(visual_tokens=sp.num_tokens, grid=[sp.grid_h, sp.grid_w])

    g = adapter.generate(inputs, seed=seed)
    new = g["token_ids"]
    n = int(new.numel())
    row.update(trace_tokens=n, truncated=g["truncated"], gen_seconds=g["seconds"],
               answer_tail=g["text"][-300:])
    if sample.get("bbox") is not None and "img_size" in sample.get("meta", {}):
        from ..data.screenspot_pro import score_click
        row["click"] = score_click(g["text"], sample["bbox"], sample["meta"]["img_size"])
    if n < min_trace:
        return {**row, "status": f"skipped: trace of {n} tokens < {min_trace}"}

    P = row["prompt_tokens"]
    steps = choose_steps(n, spread_steps, dense_len, dense_start_frac)
    full = torch.cat([inputs["input_ids"], new[None].to(inputs["input_ids"].device)], dim=1)
    kwargs = pad_like_input_ids(dict(inputs), full)
    cap = KVCapture(adapter.model, layers, (sp.start, sp.end), {k: P - 1 + s for k, s in steps.items()})
    with torch.inference_mode(), cap:
        adapter.model.model(**kwargs, use_cache=False)

    files = []
    for layer, rec in cap.results.items():
        for kind, s in steps.items():
            unit = {"dataset": dataset, "sample_id": sample["id"], "layer": layer, "kind": kind,
                    "grid": (sp.grid_h, sp.grid_w), "rope": "post", "k": rec["k"], "v": rec["v"],
                    "steps": s, "prompt_tokens": P, "trace_tokens": n, **rec[kind]}
            f = out_dir / f"{dataset}__{sample['id']}__L{layer:02d}__{kind}.pt"
            torch.save(unit, f)
            files.append(f.name)
    del cap, kwargs, full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {**row, "status": "ok", "units": len(files), "steps": {k: int(v.numel()) for k, v in steps.items()},
            "seconds": time.perf_counter() - t0}
