# 0003: Start a fresh repo; port only reusable parts of the old TPRL fork

- **Date:** 2026-11-11
- **Status:** accepted

## Context
The old repo `ashutosh-mulc/cuekeep-code` (last commit 2026-03-30) is a fork of TPRL (Cao et al.,
arXiv 2603.13394). TPRL prunes visual *embeddings* before the LLM using an RL policy (LfD then PPO)
on Qwen2.5-VL and LLaVA with transformers 4.57. CueKeep works on the *KV cache at decode time*, needs
no training, uses Qwen3-VL and transformers 5.x.

## What was ported
| Old file | New location | Change |
|---|---|---|
| `model/base_mllm.py`, `model/qwen_mllm.py` | `src/cuekeep/models/` | Rewritten for Qwen3-VL. Kept the "locate the image-token span" logic and extended it to return the merged grid (h, w) needed for g×g regions. Fixed an indentation bug in `generate_answer` |
| `data/base_loader.py`, `data/mme_loader.py` | `src/cuekeep/data/` | Seeded, deterministic splits; one sample schema |
| `data/screen_pro_loader.py` | `src/cuekeep/data/screenspot_pro.py` | Trimmed from 435 to about 80 lines; the cache-repair logic was dropped |
| `pruner/random_pruner.py` | `src/cuekeep/baselines/random_regions.py` | Now selects regions (not tokens) under the same budget interface |

## Not ported
RL networks, LfD/PPO trainers, the autoencoder, the MLP pruner, and validation scripts. These are not
relevant to a training-free decode-time method. The old repo stays available for reference.

## Consequences
The old git history is not carried over. The attribution to TPRL remains in
[third_party/README.md](../../third_party/README.md).

## Update 2026-11-15: hosting
This repo now lives in `ashutosh-mulc/cuekeep-code`. Its history was force-pushed over that repo's
`main`. The TPRL fork's old `main` (`6617257`, the pinned commit above) is kept on the branch
`archive/tprl`, so the reference copy stays available. The branch `research/cuekeep` (`9585ad2`:
the earlier Gate 0 scripts `vlm2_native.py`, `run_gate0.sh`, `RUN.md` and `TODO.md`, built on top of
the TPRL fork) was left as it is. `cuekeep-lab` refers to it, and none of those files are in this repo.
