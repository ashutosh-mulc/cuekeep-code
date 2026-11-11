#!/usr/bin/env python
"""Print the environment and run a GPU sanity check. Paste the output into the notebook when the stack changes.

    python scripts/env_check.py [--model]     # --model also loads the configured VLM's processor and config
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuekeep.tracking import env_snapshot, load_config  # noqa: E402


def gpu_checks() -> dict:
    import torch

    out = {}
    if not torch.cuda.is_available():
        return {"cuda": "NOT AVAILABLE"}
    dev = torch.device("cuda:0")
    x = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    out["bf16_matmul_ok"] = bool(torch.isfinite(x @ x).all())
    out["bf16_supported"] = torch.cuda.is_bf16_supported()
    # Pinned host → device copy bandwidth: this bounds region fetch cost (method.md, Q4).
    n = 256 * 2**20 // 2
    host = torch.empty(n, dtype=torch.bfloat16).pin_memory()
    devbuf = torch.empty(n, dtype=torch.bfloat16, device=dev)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    devbuf.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        devbuf.copy_(host, non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    out["h2d_pinned_GBps"] = round(10 * n * 2 / (start.elapsed_time(end) / 1000) / 1e9, 1)
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    q = torch.randn(1, 8, 1, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 1024, 128, device=dev, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 1, 1024, device=dev, dtype=torch.bfloat16)
    out["sdpa_float_mask_ok"] = bool(torch.isfinite(sdpa(q, k, k, attn_mask=mask)).all())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true")
    args = ap.parse_args()
    report = {"env": env_snapshot()}
    try:
        report["gpu"] = gpu_checks()
    except Exception as e:  # report, don't crash
        report["gpu"] = {"error": repr(e)}
    if args.model:
        cfg = load_config("configs/model/qwen3_vl_4b_thinking.yaml")["model"]
        from transformers import AutoConfig, AutoProcessor
        c = AutoConfig.from_pretrained(cfg["id"])
        AutoProcessor.from_pretrained(cfg["id"])
        tc = c.text_config
        report["model"] = {
            "id": cfg["id"], "layers": tc.num_hidden_layers, "heads": tc.num_attention_heads,
            "kv_heads": tc.num_key_value_heads, "head_dim": getattr(tc, "head_dim", None),
            "spatial_merge": c.vision_config.spatial_merge_size,
            "deepstack_layers": getattr(c.vision_config, "deepstack_visual_indexes", None),
        }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
