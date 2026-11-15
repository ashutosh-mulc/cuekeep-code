#!/usr/bin/env python
"""Capture KV dump units for E001 (stand-in fidelity) and E003 (attention shift).

    python scripts/dump_visual_kv.py configs/dump/gate.yaml [key=value ...] [--tag t] [--override-gates REASON]

Writes runs/DUMP-<name>/<timestamp>/ with the usual run files (config, meta, env, metrics) plus:
    units/*.pt         one file per (sample, layer, kind); format in src/cuekeep/analysis/dumps.py
    manifest.jsonl     one row per sample: trace length, truncation, visual tokens, click score, status
Pass the units directory to the analysis runs:
    python scripts/run.py experiments/E003-attention-shift/config.gate.yaml dump_path=<run dir>/units
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuekeep.analysis.dumper import dump_sample  # noqa: E402
from cuekeep.data import load_samples  # noqa: E402
from cuekeep.models import get_adapter  # noqa: E402
from cuekeep.tracking import REPO_ROOT, Run, load_config  # noqa: E402
from cuekeep.tracking.gates import check_gates, load_gates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--override-gates", default=None, metavar="REASON")
    args = ap.parse_args()

    cfg = load_config(Path(args.config).resolve(), args.overrides)
    blocking = check_gates(cfg.get("requires_gates"), load_gates())
    if blocking and not args.override_gates:
        sys.exit("[dump] held by gates that have not passed:\n  " + "\n  ".join(blocking))
    if blocking:
        cfg["gate_override"] = {"reason": args.override_gates, "blocking": blocking}

    import torch
    if str(cfg.get("device", "cuda")).startswith("cuda") and not torch.cuda.is_available():
        if not cfg.get("allow_cpu"):
            sys.exit("[dump] CUDA not available (set allow_cpu=true device=cpu model.device_map=cpu for a dry run)")
        cfg["device"] = "cpu"
    cap = cfg["capture"]
    torch.manual_seed(int(cfg.get("seed", 0)))

    with Run(f"DUMP-{cfg.get('name', 'dump')}", cfg, tag=args.tag) as run:
        units = run.dir / "units"
        units.mkdir()
        print(f"[dump] → {run.dir.relative_to(REPO_ROOT)}")
        adapter = get_adapter(cfg["model"], cfg["device"]).load()
        if adapter.model.config.text_config._attn_implementation != "sdpa":
            sys.exit("[dump] model.attn_implementation must be sdpa for the capture hook")
        n_layers = adapter.model.config.text_config.num_hidden_layers
        layers = [l for l in cap["layers"] if l < n_layers]
        if len(layers) != len(cap["layers"]):
            print(f"[dump] WARNING: model has {n_layers} layers; capturing {layers}")
        base_seed = int(cfg.get("seed", 0))
        rows = []
        with open(run.dir / "manifest.jsonl", "w") as manifest:
            for dcfg in cfg["datasets"]:
                samples = load_samples(dcfg)
                dense_n = int(cap.get("dense", {}).get("samples_per_dataset", 0))
                for j, s in enumerate(samples):
                    row = dump_sample(
                        adapter, s, seed=base_seed * 1_000_003 + j, layers=layers, out_dir=units,
                        spread_steps=int(cap["spread_steps"]),
                        dense_len=int(cap.get("dense", {}).get("length", 0)) if j < dense_n else 0,
                        dense_start_frac=float(cap.get("dense", {}).get("start_frac", 0.25)),
                        min_trace=int(cap.get("min_trace", 16)), dataset=dcfg["name"])
                    rows.append(row)
                    manifest.write(json.dumps(row, default=str) + "\n")
                    manifest.flush()
                    run.log(**{k: v for k, v in row.items() if k != "answer_tail"})
                    print(f"[dump] {dcfg['name']} {s['id']}: {row['status']}  trace={row.get('trace_tokens')}  "
                          f"vis={row.get('visual_tokens')}")

        ok = [r for r in rows if r["status"] == "ok"]
        size = sum(f.stat().st_size for f in units.glob("*.pt"))
        summary = {"samples": len(rows), "samples_ok": len(ok), "units": sum(r["units"] for r in ok),
                   "dump_gib": size / 2**30, "units_dir": str(units)}
        for ds in sorted({r["dataset"] for r in rows}):
            rs = [r for r in rows if r["dataset"] == ds and "trace_tokens" in r]
            if rs:
                summary[f"{ds}/median_trace_tokens"] = statistics.median(r["trace_tokens"] for r in rs)
                summary[f"{ds}/truncation_rate"] = sum(bool(r["truncated"]) for r in rs) / len(rs)
                summary[f"{ds}/median_visual_tokens"] = statistics.median(r["visual_tokens"] for r in rs)
            clicks = [r["click"] for r in rows if r["dataset"] == ds and "click" in r]
            if clicks:
                summary[f"{ds}/click_parsed_rate"] = sum(c["parsed"] for c in clicks) / len(clicks)
                summary[f"{ds}/click_hit_rel1000"] = sum(c["hit_rel1000"] for c in clicks) / len(clicks)
                summary[f"{ds}/click_hit_pixels"] = sum(c["hit_pixels"] for c in clicks) / len(clicks)
        run.finish(summary)
        print(f"[dump] done: {summary}")


if __name__ == "__main__":
    main()
