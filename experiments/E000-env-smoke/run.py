"""E000: load the model, generate on a few samples, and record trace lengths, visual-token counts and memory."""
from __future__ import annotations

from cuekeep.data import load_samples
from cuekeep.eval import GenerationRecord, Summary, cuda_peak_memory
from cuekeep.models import get_adapter


def main(cfg: dict, run) -> dict:
    adapter = get_adapter(cfg["model"], cfg["device"]).load()
    summaries = {}
    for dcfg in cfg["datasets"]:
        summary = Summary()
        for s in load_samples(dcfg):
            inputs = adapter.build_inputs(s["image"], s["question"])
            spans = adapter.visual_spans(inputs)
            with cuda_peak_memory() as mem:
                out = adapter.generate(inputs)
            rec = GenerationRecord(s["id"], None, out["num_new_tokens"], out["seconds"], peak_mem_bytes=mem["peak_bytes"])
            summary.add(rec)
            run.log(dataset=dcfg["name"], sample=s["id"], prompt_tokens=int(inputs["input_ids"].shape[1]),
                    visual_tokens=sum(sp.num_tokens for sp in spans),
                    grids=[(sp.grid_h, sp.grid_w) for sp in spans],
                    trace_tokens=out["num_new_tokens"], seconds=out["seconds"],
                    ms_per_token=rec.ms_per_token, peak_mem_gib=mem["peak_bytes"] / 2**30,
                    answer_head=out["text"][-200:])
        summaries[dcfg["name"]] = summary.as_dict()
    return {f"{d}/{k}": v for d, m in summaries.items() for k, v in m.items()}
