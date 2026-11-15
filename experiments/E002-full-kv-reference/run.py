"""E002: full-KV reference accuracy, trace length and timing (gate G6, then the main tier).

In-repo harness (decisions/0006): each benchmark sample is generated with the model's own decoding
and a per-sample seed, and scored by rule (`cuekeep.eval.scoring`). Benchmarks and their official
Qwen3-VL-4B-Thinking scores are listed in the config.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Dict, List

from cuekeep.data import load_samples, split_indices
from cuekeep.eval import GenerationRecord, Summary
from cuekeep.eval.scoring import score_sample
from cuekeep.models import get_adapter


def main(cfg: dict, run) -> dict:
    adapter = get_adapter(cfg["model"], cfg["device"]).load()
    kvb = adapter.kv_bytes_per_token()
    out: Dict[str, object] = {}
    est_hours = 0.0
    all_rows: List[dict] = []
    with open(Path(run.dir) / "samples.jsonl", "w") as fh:
        for bcfg in cfg["benchmarks"]:
            name = bcfg["name"]
            dev = set(split_indices(bcfg["full_size"], "dev", bcfg["dev_size"])) if bcfg.get("full_size") else set()
            samples = load_samples(bcfg)
            for seed in cfg.get("seeds", [cfg.get("seed", 0)]):
                summary = Summary()
                test_correct = []
                for j, s in enumerate(samples):
                    inputs = adapter.build_inputs(s["image"], s["question"], s.get("system"))
                    spans = adapter.visual_spans(inputs)
                    g = adapter.generate(inputs, seed=int(seed) * 1_000_003 + j)
                    sc = score_sample(s, g["text"])
                    n_vis = sum(sp.num_tokens for sp in spans)
                    rec = GenerationRecord(s["id"], bool(sc["correct"]), g["num_new_tokens"], g["seconds"],
                                           prefill_seconds=g["prefill_seconds"], truncated=g["truncated"],
                                           prompt_tokens=int(inputs["input_ids"].shape[1]), visual_tokens=n_vis,
                                           kv_bytes_per_token=kvb, visual_kv_gpu_bytes=n_vis * kvb,
                                           prefill_peak_bytes=g["prefill_peak_bytes"],
                                           decode_peak_bytes=g["decode_peak_bytes"])
                    summary.add(rec)
                    idx = s.get("meta", {}).get("index")
                    in_dev = idx in dev
                    if not in_dev:
                        test_correct.append(bool(sc["correct"]))
                    row = {"benchmark": name, "seed": seed, "sample": s["id"], "index": idx, "in_dev": in_dev,
                           "correct": bool(sc["correct"]), "pred": str(sc.get("pred"))[:200],
                           "answer": str(s.get("answer"))[:200], "trace_tokens": rec.num_new_tokens,
                           "truncated": rec.truncated, "seconds": rec.seconds,
                           "decode_ms_per_token": rec.ms_per_token, "visual_tokens": n_vis,
                           **{k: v for k, v in sc.items() if k not in ("pred", "correct")},
                           "answer_tail": g["text"][-400:]}
                    fh.write(json.dumps(row, default=str) + "\n")
                    fh.flush()
                    run.log(**{k: v for k, v in row.items() if k != "answer_tail"})
                    all_rows.append(row)
                d = summary.as_dict()
                acc, n = d["accuracy"], d["n"]
                out.update({f"{name}/s{seed}/{k}": v for k, v in d.items()})
                out[f"{name}/s{seed}/accuracy_se"] = math.sqrt(acc * (1 - acc) / n) if n else float("nan")
                if test_correct:
                    out[f"{name}/s{seed}/accuracy_test_split"] = sum(test_correct) / len(test_correct)
                    out[f"{name}/s{seed}/n_test_split"] = len(test_correct)
            accs = [out[f"{name}/s{s}/accuracy"] for s in cfg.get("seeds", [cfg.get("seed", 0)])]
            out[f"{name}/accuracy_mean"] = statistics.fmean(accs)
            out[f"{name}/accuracy_std_over_seeds"] = statistics.pstdev(accs) if len(accs) > 1 else 0.0
            rows = [r for r in all_rows if r["benchmark"] == name]
            sec_per_sample = statistics.fmean(r["seconds"] for r in rows)
            if bcfg.get("full_size"):
                est_hours += bcfg["full_size"] * len(cfg.get("main_seeds", [0, 1, 2])) * sec_per_sample / 3600
            out[f"{name}/seconds_per_sample_mean"] = sec_per_sample

    out["main_tier_gpu_hours_estimate"] = est_hours
    gc = cfg.get("gate")
    if gc:
        ran, close, trunc = True, True, True
        for bcfg in cfg["benchmarks"]:
            name, seed0 = bcfg["name"], cfg.get("seeds", [cfg.get("seed", 0)])[0]
            n = out.get(f"{name}/s{seed0}/n", 0)
            ran = ran and n >= bcfg.get("limit", n)
            acc = 100 * out.get(f"{name}/accuracy_mean", 0.0)
            if bcfg.get("official") is not None:
                gap = bcfg["official"] - acc
                out[f"gate/{name}/points_below_official"] = gap
                close = close and gap <= gc["max_points_below_official"]
            tr = out.get(f"{name}/s{seed0}/truncation_rate", 1.0)
            trunc = trunc and tr < gc["max_truncation_rate"]
        out.update({"gate/c1_all_ran": ran, "gate/c2_near_official": close, "gate/c3_truncation": trunc,
                    "gate/c4_cost_estimate_hours": est_hours, "gate/pass": ran and close and trunc})
    return out
