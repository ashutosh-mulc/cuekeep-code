"""The KV dump format, shared by E001 (stand-in fidelity) and E003 (attention shift).

One dump captures everything both gates need, and it is small. Visual KV does not change during
decoding, so it is stored once per (sample, layer). The text and generated keys enter only through
their summed contribution to each query's attention, so each sampled decode step stores a
log-sum-exp and one mean value vector instead of the text KV (see `pooling.summarize_other`).

A dump is a directory of `.pt` files (or a single file). Each file holds one unit dict or a list of them:

    sample_id : str            dataset : str             layer : int
    grid      : (h, w)         merged-token grid of the (single) image
    rope      : "post" | "pre" whether k was captured after or before rotary embedding
    k, v      : [kvh, n_vis, d]            visual KV at this layer (bf16 is fine)
    steps     : LongTensor [S]             decode-step index of each sampled query
    q         : [heads, S, d]              decode queries, post-norm and post-RoPE
    other_lse : [heads, S]                 logsumexp of scale·q·k over all non-visual keys visible at that step
    other_out : [heads, S, d]              their attention-weighted mean value (same scale, softmax over those keys)
    attn_out  : [heads, S, d]  (optional)  the model's true per-head attention output, for the capture check

Size: about 4 KB per visual token per (sample, layer) in bf16 for Qwen3-VL-4B (8 KV heads × 128 × K+V),
plus about 16 KB per sampled step. 8,000 visual tokens and 64 steps come to about 34 MB per unit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Union

import torch

from ..cache.pooling import summarize_other


def iter_dump(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Yield raw unit dicts from a dump directory (files in name order) or a single file, one file at a time."""
    path = Path(path)
    files = sorted(path.glob("*.pt")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no .pt files under {path}")
    for f in files:
        data = torch.load(f, map_location="cpu")
        for u in (data if isinstance(data, list) else [data]):
            yield u


def to_unit(u: Dict[str, Any], device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    """Normalise a raw unit to the grouped form the experiments use.

    Output: k, v [kvh, n, d]; q [kvh, Q, d]; other_lse [kvh, Q] or None; other_out [kvh, Q, d] or None;
    grid; steps (S, the number of decode steps); stride (decode steps between samples; 1 = dense window); name. With GQA, query head h belongs to KV head h // grp,
    so [heads, S, ·] reshapes to [kvh, grp·S, ·], and the decode step of grouped query j is j % S.

    Units that carry text KV (`text_k`, `text_v`, with q already grouped [kvh, Q, d]), as the synthetic
    generator produces, are summarised here, so both sources share one code path.
    """
    k, v = u["k"].to(device).float(), u["v"].to(device).float()
    kvh, _, d = k.shape
    q = u["q"].to(device).float()
    scale = d ** -0.5
    if u.get("text_k") is not None:                      # synthetic / legacy: text KV given, q grouped
        lse, out = summarize_other(q, u["text_k"].to(device), u["text_v"].to(device), scale)
        steps = int(u.get("steps_per_group", q.shape[1]))
    elif u.get("other_lse") is not None:                 # dump: per-head, per-step summaries
        heads, S, _ = q.shape
        if heads % kvh:
            raise ValueError(f"{heads} query heads do not divide into {kvh} KV heads")
        q = q.reshape(kvh, -1, d)
        lse = u["other_lse"].to(device).float().reshape(kvh, -1)
        out = u["other_out"].to(device).float().reshape(kvh, -1, u["other_out"].shape[-1])
        steps = S
    else:                                                # visual-only unit
        lse = out = None
        steps = int(u.get("steps_per_group", q.shape[1]))
    st = u.get("steps")
    stride = int(torch.as_tensor(st).diff().float().median()) if st is not None and len(st) > 1 else None
    name = u.get("name") or (f"{u.get('dataset', '?')}/{u.get('sample_id', '?')}/L{u.get('layer', '?')}"
                             f"/{u.get('rope', 'post')}/{'dense' if stride == 1 else 'spread'}")
    return {"k": k, "v": v, "q": q, "other_lse": lse, "other_out": out, "grid": tuple(u["grid"]),
            "steps": steps, "stride": stride, "name": name, "layer": u.get("layer"), "dataset": u.get("dataset"),
            "sample_id": u.get("sample_id")}
