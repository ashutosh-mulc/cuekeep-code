"""E004: paging fidelity under teacher forcing (gate G4). Method and criteria: this folder's README.

Per sample:
1. Generate a full-KV trace (the model's own decoding, per-sample seed); keep its first `trace_tokens`.
2. Teacher-forced forwards over prompt + trace:
     base           the untouched model (full KV, the model's own attention kernel)
     all_resident   full KV through the simulation's float32 path: the reference for every comparison,
                    so differences between conditions come from paging only, not from kernel numerics
   then, for each region size g and budget f: cuekeep, oracle, random_regions, standins_only, and
   prune_once (once per f; it does not depend on g).
3. Next-token KL(reference ‖ condition) and top-1 agreement at the decode positions.
The hook check (criterion 1) compares all_resident with base.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List

import torch

from cuekeep.analysis.capture import pad_like_input_ids
from cuekeep.cache import region_ids_for_grid
from cuekeep.data import load_samples
from cuekeep.models import get_adapter
from cuekeep.sim.paged_sim import (
    AllResident, CueKeep, Oracle, PagingSimulation, PruneOnce, RandomRegions, StandinsOnly, compare_next_token,
)


def paged_layers(spec, n_layers: int) -> List[int]:
    """Decoder layers to page: a Python-slice string ("2:" → [2, …, n-1]) or an explicit list of ints."""
    if isinstance(spec, (list, tuple)):
        return [int(l) for l in spec if 0 <= int(l) < n_layers]
    if not isinstance(spec, str):
        raise ValueError(f"layers_paged must be a slice string like \"2:\" (quoted in YAML) or a list, got {spec!r}")
    parts = [int(x) if x else None for x in spec.split(":")]
    return list(range(n_layers))[slice(*parts)]


def _hidden(model, kwargs, P: int, L: int) -> torch.Tensor:
    with torch.inference_mode():
        h = model.model(**kwargs, use_cache=False).last_hidden_state[0]
    return h[P:L - 1]                                    # position P+t predicts trace token t+1


def _summ(cmp: Dict[str, torch.Tensor]) -> Dict[str, float]:
    kl, top1 = cmp["kl"].double(), cmp["top1"].float()
    n = kl.numel()
    thirds = [slice(0, n // 3), slice(n // 3, 2 * n // 3), slice(2 * n // 3, n)]
    out = {"positions": n, "kl_mean": float(kl.mean()), "kl_median": float(kl.median()),
           "top1": float(top1.mean())}
    for name, s in zip(("early", "mid", "late"), thirds):
        if kl[s].numel():
            out[f"kl_mean_{name}"] = float(kl[s].mean())
    return out


def run_sample(adapter, sample, seed, cfg, run, dataset) -> List[dict]:
    sw, method = cfg["sweep"], cfg["method"]
    model = adapter.model
    inputs = adapter.build_inputs(sample["image"], sample["question"], sample.get("system"))
    spans = adapter.visual_spans(inputs)
    if len(spans) != 1:
        return [{"dataset": dataset, "sample_id": sample["id"], "status": f"skipped: {len(spans)} images"}]
    sp = spans[0]
    g_out = adapter.generate(inputs, seed=seed)
    new = g_out["token_ids"][: int(cfg["trace_tokens"])]
    n = int(new.numel())
    base_row = {"dataset": dataset, "sample_id": sample["id"], "seed": seed, "trace_tokens": n,
                "truncated_generation": g_out["truncated"], "visual_tokens": sp.num_tokens}
    if n < int(cfg.get("min_trace", 32)):
        return [{**base_row, "status": f"skipped: trace of {n} tokens"}]
    P = int(inputs["input_ids"].shape[1])
    L = P + n
    kwargs = pad_like_input_ids(dict(inputs), torch.cat([inputs["input_ids"], new[None].to(inputs["input_ids"].device)], 1))
    layers = paged_layers(method.get("layers_paged", "2:"), model.config.text_config.num_hidden_layers)
    lm_head = model.lm_head
    rows = []

    def simulate(policy, g):
        rid = region_ids_for_grid(sp.grid_h, sp.grid_w, g)
        sim = PagingSimulation(model, layers, (sp.start, sp.end), P, rid, policy, chunk=int(cfg.get("chunk", 128)))
        with sim:
            h = _hidden(model, kwargs, P, L)
        st = list(sim.stats.values())
        agg = {k: statistics.fmean(s[k] for s in st) for k in (st[0] if st else {}) if isinstance(st[0][k], (int, float))}
        return h, agg

    h_base = _hidden(model, kwargs, P, L)
    g0 = sw["region_size"][0]
    h_ref, _ = simulate(AllResident(), g0)
    eq = _summ(compare_next_token(h_base, h_ref, lm_head))
    rows.append({**base_row, "status": "ok", "condition": "all_resident_vs_base", "g": None, "budget": 1.0, **eq})
    run.log(**rows[-1])

    for f in sw["budgets"]:
        h, st = simulate(PruneOnce(f), g0)
        rows.append({**base_row, "status": "ok", "condition": "prune_once", "g": None, "budget": f,
                     **_summ(compare_next_token(h_ref, h, lm_head)), **{f"stat_{k}": v for k, v in st.items()}})
        run.log(**rows[-1])
        for g in sw["region_size"]:
            for name in sw["conditions"]:
                if name == "prune_once":
                    continue
                policy = {"cuekeep": lambda: CueKeep(f, method), "oracle": lambda: Oracle(f),
                          "random_regions": lambda: RandomRegions(f, seed),
                          "standins_only": lambda: StandinsOnly()}[name]()
                if name == "standins_only" and f != sw["budgets"][0]:
                    continue                                 # independent of the budget
                h, st = simulate(policy, g)
                rows.append({**base_row, "status": "ok", "condition": name, "g": g, "budget": f,
                             **_summ(compare_next_token(h_ref, h, lm_head)), **{f"stat_{k}": v for k, v in st.items()}})
                run.log(**rows[-1])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _median(rows, key):
    vals = [r[key] for r in rows if key in r]
    return statistics.median(vals) if vals else None


def gate_verdict(rows: List[dict], gc: dict) -> Dict[str, object]:
    ok = [r for r in rows if r.get("status") == "ok"]
    out: Dict[str, object] = {}
    eq = [r for r in ok if r["condition"] == "all_resident_vs_base"]
    n = sum(r["positions"] for r in eq)
    top1 = sum(r["top1"] * r["positions"] for r in eq) / max(n, 1)
    kl = sum(r["kl_mean"] * r["positions"] for r in eq) / max(n, 1)
    out.update({"gate/equivalence_top1": top1, "gate/equivalence_kl_mean": kl})
    c1 = bool(eq) and top1 >= gc["equivalence_min_top1"] and kl <= gc["equivalence_max_kl"]

    g, f = gc["region_size"], gc["budget"]
    ratios, top1s = {}, {}
    for ds in sorted({r["dataset"] for r in ok}):
        ck = [r for r in ok if r["dataset"] == ds and r["condition"] == "cuekeep" and r["g"] == g and r["budget"] == f]
        po = [r for r in ok if r["dataset"] == ds and r["condition"] == "prune_once" and r["budget"] == f]
        if ck and po:
            ratios[ds] = _median(ck, "kl_mean") / max(_median(po, "kl_mean"), 1e-12)
            pos = sum(r["positions"] for r in ck)
            top1s[ds] = sum(r["top1"] * r["positions"] for r in ck) / max(pos, 1)
            out[f"gate/{ds}/cuekeep_kl_median"] = _median(ck, "kl_mean")
            out[f"gate/{ds}/prune_once_kl_median"] = _median(po, "kl_mean")
            out[f"gate/{ds}/kl_ratio_cuekeep_over_prune_once"] = ratios[ds]
            out[f"gate/{ds}/cuekeep_top1"] = top1s[ds]
    c2 = bool(ratios) and min(ratios.values()) <= gc["max_kl_ratio_vs_prune_once"] and max(ratios.values()) <= 1.0
    c3 = bool(top1s) and min(top1s.values()) >= gc["min_top1"]
    out.update({"gate/c1_equivalence": c1, "gate/c2_kl_vs_prune_once": c2, "gate/c3_top1": c3,
                "gate/pass": c1 and c2 and c3})
    return out


def main(cfg: dict, run) -> dict:
    adapter = get_adapter(cfg["model"], cfg["device"]).load()
    if adapter.model.config.text_config._attn_implementation != "sdpa":
        raise ValueError("model.attn_implementation must be sdpa for the simulation hook")
    base_seed = int(cfg.get("seed", 0))
    rows: List[dict] = []
    with open(Path(run.dir) / "rows.jsonl", "w") as fh:
        for dcfg in cfg["datasets"]:
            for j, s in enumerate(load_samples(dcfg)):
                for r in run_sample(adapter, s, base_seed * 1_000_003 + j, cfg, run, dcfg["name"]):
                    rows.append(r)
                    fh.write(json.dumps(r, default=str) + "\n")
                    fh.flush()
                print(f"[E004] {dcfg['name']} {s['id']}: {rows[-1].get('status')}")

    ok = [r for r in rows if r.get("status") == "ok"]
    summary: Dict[str, object] = {"samples": len({(r["dataset"], r["sample_id"]) for r in rows}),
                                  "samples_ok": len({(r["dataset"], r["sample_id"]) for r in ok})}
    keys = sorted({(r["dataset"], r["condition"], r["g"], r["budget"]) for r in ok}, key=str)
    for ds, cond, g, f in keys:
        sel = [r for r in ok if (r["dataset"], r["condition"], r["g"], r["budget"]) == (ds, cond, g, f)]
        tag = f"{ds}/{cond}/g{g}/b{f}"
        summary[f"{tag}/kl_mean_median"] = _median(sel, "kl_mean")
        summary[f"{tag}/top1_mean"] = statistics.fmean(r["top1"] for r in sel)
    if cfg.get("gate"):
        summary.update(gate_verdict(rows, cfg["gate"]))
    return summary
