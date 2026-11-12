"""E000: load the model, generate on a few samples, and record trace lengths, visual-token counts,
per-phase time and memory, and the KV byte breakdown. Also checks the token-order assumption."""
from __future__ import annotations

import statistics

import torch

from cuekeep.data import load_samples
from cuekeep.eval import GenerationRecord, Summary
from cuekeep.models import get_adapter


def token_order_check(adapter) -> dict:
    """Is the merged-token sequence row-major over the merged grid? (region_ids_for_grid assumes so.)

    Feed an image whose left half is white and right half black. Reshape the visual features
    row-major to [h, w, D] and measure how well the left and right halves separate. Then compare with
    the separation obtained if the sequence were column-major. Row-major should separate far better.
    """
    from PIL import Image

    img = Image.new("RGB", (448, 336), "black")
    img.paste((255, 255, 255), (0, 0, 224, 336))
    try:
        inputs = adapter.build_inputs(img, "Describe the image.")
        spans = adapter.visual_spans(inputs)
        feats = adapter.model.get_image_features(pixel_values=inputs["pixel_values"],
                                                 image_grid_thw=inputs["image_grid_thw"])
        while isinstance(feats, (tuple, list)) or hasattr(feats, "pooler_output"):
            feats = feats[0] if isinstance(feats, (tuple, list)) else feats.pooler_output
        f = feats.float().reshape(-1, feats.shape[-1])
        h, w = spans[0].grid_h, spans[0].grid_w

        def separation(grid: torch.Tensor) -> float:
            left, right = grid[:, : w // 2].reshape(-1, grid.shape[-1]), grid[:, w // 2:].reshape(-1, grid.shape[-1])
            between = (left.mean(0) - right.mean(0)).norm()
            within = 0.5 * (left.std(0).norm() + right.std(0).norm())
            return float(between / within.clamp_min(1e-6))

        row = separation(f.reshape(h, w, -1))
        col = separation(f.reshape(w, h, -1).transpose(0, 1))
        return {"grid": [h, w], "sep_row_major": row, "sep_col_major": col, "row_major_ok": row > 2 * col}
    except Exception as e:  # record, don't fail the smoke test
        return {"error": repr(e), "row_major_ok": None}


def main(cfg: dict, run) -> dict:
    adapter = get_adapter(cfg["model"], cfg["device"]).load()
    kvb, wbytes = adapter.kv_bytes_per_token(), adapter.weight_bytes()
    order = token_order_check(adapter)
    run.log(event="setup", kv_bytes_per_token=kvb, weight_gib=wbytes / 2**30, token_order=order)

    base_seed = int(cfg.get("seed", 0))
    out_summary = {"kv_bytes_per_token": kvb, "weight_gib": wbytes / 2**30, "token_order_row_major": order["row_major_ok"]}
    medians = {}
    for dcfg in cfg["datasets"]:
        summary = Summary()
        try:
            samples = load_samples(dcfg)
        except Exception as e:   # e.g. unpinned ScreenSpot-Pro format: log it, keep the other datasets
            run.log(dataset=dcfg["name"], load_error=repr(e))
            out_summary[f"{dcfg['name']}/load_error"] = repr(e)
            continue
        for j, s in enumerate(samples):
            inputs = adapter.build_inputs(s["image"], s["question"])
            spans = adapter.visual_spans(inputs)
            n_vis = sum(sp.num_tokens for sp in spans)
            g = adapter.generate(inputs, seed=base_seed * 1_000_003 + j)   # per-sample seed, stable across methods
            rec = GenerationRecord(
                s["id"], None, g["num_new_tokens"], g["seconds"], prefill_seconds=g["prefill_seconds"],
                truncated=g["truncated"], prompt_tokens=int(inputs["input_ids"].shape[1]), visual_tokens=n_vis,
                kv_bytes_per_token=kvb, visual_kv_gpu_bytes=n_vis * kvb,          # full KV: all visual KV stays on GPU
                prefill_peak_bytes=g["prefill_peak_bytes"], decode_peak_bytes=g["decode_peak_bytes"],
            )
            summary.add(rec)
            run.log(dataset=dcfg["name"], sample=s["id"], seed=base_seed * 1_000_003 + j,
                    prompt_tokens=rec.prompt_tokens, visual_tokens=n_vis,
                    grids=[(sp.grid_h, sp.grid_w) for sp in spans],
                    trace_tokens=rec.num_new_tokens, truncated=rec.truncated,
                    seconds=rec.seconds, prefill_seconds=rec.prefill_seconds,
                    decode_ms_per_token=rec.ms_per_token,
                    visual_kv_mib=rec.visual_kv_bytes / 2**20, total_kv_end_mib=rec.total_kv_bytes_end / 2**20,
                    prefill_peak_gib=(rec.prefill_peak_bytes or 0) / 2**30,
                    decode_peak_gib=(rec.decode_peak_bytes or 0) / 2**30,
                    answer_tail=g["text"][-300:])
        out_summary.update({f"{dcfg['name']}/{k}": v for k, v in summary.as_dict().items()})
        rs = summary.records
        if rs:
            medians[dcfg["name"]] = {
                "trace": statistics.median(r.num_new_tokens for r in rs),
                "visual_frac": statistics.median(r.visual_tokens / max(r.prompt_tokens, 1) for r in rs),
                "visual_kv_share_end": statistics.median(r.visual_kv_bytes / max(r.total_kv_bytes_end, 1) for r in rs),
                "truncation_rate": sum(r.truncated for r in rs) / len(rs),
            }
            out_summary.update({f"{dcfg['name']}/median_{k}": v for k, v in medians[dcfg["name"]].items() if k != "truncation_rate"})
    if cfg.get("gate"):
        out_summary.update(gate_verdict(medians, order["row_major_ok"], cfg["gate"]))
    return out_summary


def gate_verdict(medians: dict, row_major, gc: dict) -> dict:
    """G1 pass criteria (research/experiment_design.md)."""
    vis = medians.get(gc["visual_dataset"])
    c1 = row_major is True
    c2 = any(m["trace"] >= gc["min_median_trace_tokens"] for m in medians.values())
    c3 = vis is not None and vis["visual_frac"] >= gc["min_median_visual_frac"]
    truncated = any(m["truncation_rate"] > 0 for m in medians.values())
    return {
        "gate/c1_token_order": c1, "gate/c2_long_traces": c2, "gate/c3_visual_share": c3,
        "gate/all_datasets_loaded": vis is not None,
        "gate/needs_full_length_rerun": truncated,       # trace medians are censored until rerun at 40960
        "gate/pass": bool(c1 and c2 and c3 and not truncated),
    }
