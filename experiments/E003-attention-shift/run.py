"""E003: does the set of regions a thinking trace attends to shift during decoding? (H2, Q3)

Reads the same dump as E001. For each post-RoPE unit, region size g and budget fraction f, it
computes the exact per-step region attention mass (true keys, true queries, text summary) and then
`cuekeep.analysis.shift_metrics`: how much visual attention mass a *static* top-f set (chosen from
the first steps), a *lagged* set (previous sampled step) and a *dynamic* set (oracle) each cover.

Units with stride 1 are dense windows: there, "lagged" is exactly a 1-step lookahead.
The gate verdict (G3) is computed into summary["gate/..."] when the config has a `gate:` block.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List

from cuekeep.analysis import iter_dump, per_step_mass, shift_metrics, to_unit
from cuekeep.cache import exact_region_log_mass, region_ids_for_grid, region_log_shares


def unit_rows(u: dict, raw: dict, cfg: dict) -> List[dict]:
    k, q = u["k"], u["q"]
    h, w = u["grid"]
    scale = q.shape[-1] ** -0.5
    rows = []
    for g in cfg["region_size"]:
        rid = region_ids_for_grid(h, w, g, device=k.device)
        R = int(rid.max()) + 1
        shares = region_log_shares(exact_region_log_mass(q, k, rid, R, scale), u["other_lse"])
        mass = per_step_mass(shares, u["steps"]).cpu()
        for f in cfg["budgets"]:
            rows.append({"unit": u["name"], "dataset": u["dataset"], "sample_id": u["sample_id"],
                         "layer": u["layer"], "stride": u["stride"], "window": u["kind"],
                         "g": g, "budget": f, **shift_metrics(mass, f, cfg.get("warmup_frac", 0.05))})
    return rows


def gate_verdict(rows: List[dict], gc: dict) -> Dict[str, object]:
    """G3 pass criteria (research/experiment_design.md)."""
    sel = [r for r in rows if r["g"] == gc["region_size"] and r["budget"] == gc["budget"]]
    out: Dict[str, object] = {}
    gaps = {}
    for ds in sorted({r["dataset"] for r in sel}):
        spread = [r["gap_dynamic_static"] for r in sel if r["dataset"] == ds and r["window"] == "spread"]
        if spread:
            gaps[ds] = statistics.median(spread)
            out[f"gate/gap_dynamic_static/{ds}"] = gaps[ds]
    dense = [r["gap_dynamic_lagged"] for r in sel if r["window"] == "dense"]
    out["gate/lag1_gap_dense"] = statistics.median(dense) if dense else None
    best = max(gaps.values()) if gaps else float("nan")
    c1 = bool(gaps) and best >= gc["min_gap"]
    c2 = bool(dense) and statistics.median(dense) <= gc["max_lag1_gap"]
    out.update({
        "gate/c1_attention_shifts": c1,
        "gate/c2_lag1_tracks": c2,
        "gate/marginal": bool(gaps) and gc["kill_gap"] <= best < gc["min_gap"],     # needs the human
        "gate/kill": bool(gaps) and best < gc["kill_gap"],                          # recall has no edge over prune-once
        "gate/pass": c1 and c2,
    })
    return out


def main(cfg: dict, run) -> dict:
    if not cfg.get("dump_path"):
        raise ValueError("needs dump_path=<directory written by scripts/dump_visual_kv.py>")
    rows: List[dict] = []
    for raw in iter_dump(cfg["dump_path"]):
        if raw.get("rope", "post") != "post" or raw.get("other_lse") is None:
            continue
        for row in unit_rows(to_unit(raw, cfg["device"]), raw, cfg):
            rows.append(row)
            run.log(**row)
    if not rows:
        raise RuntimeError("no post-RoPE units with text summaries in the dump")

    summary: Dict[str, object] = {"units": len({r["unit"] for r in rows})}
    metrics = ("cov_static", "cov_lagged", "cov_dynamic", "gap_dynamic_static", "gap_dynamic_lagged",
               "turnover", "late_new_frac", "visual_mass_early", "visual_mass_late")
    for window in sorted({r["window"] for r in rows}):
        for g in cfg["region_size"]:
            for f in cfg["budgets"]:
                sel = [r for r in rows if (r["window"], r["g"], r["budget"]) == (window, g, f)]
                for m in metrics:
                    if sel:
                        summary[f"{window}/g{g}/b{f}/{m}"] = statistics.median(r[m] for r in sel)
    if cfg.get("gate"):
        summary.update(gate_verdict(rows, cfg["gate"]))
    (Path(run.dir) / "rows.json").write_text(json.dumps(rows, indent=1))
    return summary
