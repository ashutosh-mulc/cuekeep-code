"""Build a tiny, randomly initialised Qwen3-VL for CPU integration tests.

It keeps everything the pipeline depends on from the real model: the architecture classes, the
tokenizer and chat template, the image processor (patch 16, spatial merge 2, so 32 px per merged
token), special token ids, M-RoPE, DeepStack and GQA. Only the sizes shrink: 4 text layers, hidden
64, 4 query heads / 2 KV heads of dim 16, and a 2-block vision tower. The weights are random, so its
outputs are meaningless; the tests check that the code runs and that captured tensors are consistent.

    python -m cuekeep.testing.tiny_qwen3vl <out_dir> [--source Qwen/Qwen3-VL-4B-Thinking]

Needs the source model's config and processor files (a few MB; no weights), from the HF cache or the Hub.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SOURCE = "Qwen/Qwen3-VL-4B-Thinking"

TEXT = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16)
VISION = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
              deepstack_visual_indexes=[0])
MROPE_SECTION = [2, 3, 3]          # must sum to head_dim / 2


def _set_mrope(text_cfg) -> None:
    """Shrink mrope_section to the tiny head_dim (transformers 5.x: rope_parameters; older: rope_scaling)."""
    attr = "rope_parameters" if isinstance(getattr(text_cfg, "rope_parameters", None), dict) else "rope_scaling"
    rp = dict(getattr(text_cfg, attr))
    rp["mrope_section"] = list(MROPE_SECTION)
    setattr(text_cfg, attr, rp)


def build(out_dir: str | Path, source: str = SOURCE, seed: int = 0) -> Path:
    import torch
    from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

    out = Path(out_dir)
    if (out / "config.json").exists() and (out / "model.safetensors").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(source)
    for k, v in TEXT.items():
        setattr(cfg.text_config, k, v)
    for k, v in VISION.items():
        setattr(cfg.vision_config, k, v)
    _set_mrope(cfg.text_config)
    torch.manual_seed(seed)
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    try:
        model.generation_config = GenerationConfig.from_pretrained(source)
    except Exception:
        pass
    model.save_pretrained(out)
    AutoProcessor.from_pretrained(source).save_pretrained(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.force and Path(a.out_dir).exists():
        shutil.rmtree(a.out_dir)
    print(build(a.out_dir, a.source))


if __name__ == "__main__":
    main()
