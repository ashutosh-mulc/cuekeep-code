# CueKeep — Pooled-KV Paging for Visual Tokens in Thinking VLMs

**Status:** active research · **Target venue:** ICML 2027 · **Primary model:** Qwen3-VL-4B-Thinking

Thinking VLMs produce traces of thousands of tokens while they keep every visual token's KV
resident on the GPU. CueKeep keeps a **cheap pooled stand-in** for every image region on the GPU,
**pages the exact region KV to host memory**, and **recalls a region on demand** when the
decoding query needs it. The whole method is training-free.

```
image ─► ViT ─► merger ─► LLM prefill ─► PagedVisualCache
                                         ├─ text KV            (dense, GPU)
                                         ├─ pooled region K/V  (+ log m bias, GPU, always attended)
                                         ├─ resident slot pool (exact KV of hot regions, GPU, hard cap)
                                         └─ host store         (exact KV of all regions, pinned CPU)
decode step t:  q_t ─► demand score per region ─► fetch top regions (H2D) / evict under cap
```

The method is defined in [research/method.md](research/method.md). The novelty claim against
ViMoD, VisionPulse, SparseVILA and ShadowKV is set out in [research/related_work.md](research/related_work.md).

## Repository map

| Path | What lives there |
|---|---|
| [research/](research/) | The research record: overview, method, related work, roadmap, decision log, dated lab notebook |
| [experiments/](experiments/) | One folder per experiment (`E###-slug/`), each with a small **gate** tier that holds back its main tier. Pre-registered thresholds, runs, results and gate decisions. Index: [experiments/README.md](experiments/README.md); design: [research/experiment_design.md](research/experiment_design.md); ledger: [experiments/gates.yaml](experiments/gates.yaml) |
| [configs/](configs/) | Composable YAML configs (`model/`, `method/`, `eval/`); every experiment config extends `configs/base.yaml` |
| [src/cuekeep/](src/cuekeep/) | The library: `cache/` (the paged cache and pooling math), `analysis/` (dump format, attention-shift metrics), `models/`, `data/`, `baselines/`, `eval/`, `tracking/` (runs, configs, gates) |
| [paper/](paper/) | ARS `academic-pipeline` workspace: one folder per stage (research → draft → integrity → review → revision → final), the Material Passport, figures |
| [scripts/](scripts/) | `new_experiment.py`, `run.py`, `gate.py`, `launch.sh` / `slurm_template.sbatch`, `setup_server.sh`, `env_check.py`, `export_provenance.py`, `build_corpus.sh` |
| [tests/](tests/) | Unit tests; the pooling-math tests run on CPU |
| [third_party/](third_party/) | External codebases we borrow from: provenance, licenses, pinned commits |
| [legacy/](legacy/) | Code from the earlier pruning direction, kept for reference and outside the gated pipeline: [vlm2_gate0/](legacy/vlm2_gate0/) (unpruned Qwen2.5-VL baselines on VLM²-Bench, and that direction's notes) |
| `runs/` | Raw run outputs (gitignored). Curated results are copied into `experiments/E###/results/` |

## Workflow

On the GPU server (NVIDIA, CUDA). The full ordered task list for the server agent is
[SERVER_CHECKLIST.md](SERVER_CHECKLIST.md).

```bash
bash scripts/setup_server.sh                # .venv with CUDA torch, transformers 5.x, VLMEvalKit
python scripts/env_check.py --model         # GPUs, versions, H2D bandwidth, model shapes → paste into the notebook
GPUS=0 bash scripts/launch.sh experiments/E000-env-smoke/config.gate.yaml     # detached, logged
```

Anywhere:

```bash
python scripts/new_experiment.py "region size sweep"     # scaffolds experiments/E00N-region-size-sweep/
python scripts/run.py experiments/E001-pooled-attention-fidelity/config.synthetic.yaml sweep.top_b=4 --tag quick   # CPU code check
python scripts/gate.py list                 # which gates have passed; a held experiment will not start
pytest -q                                   # CPU tests (needs torch + pyyaml)
```

Every run writes `runs/<exp-id>/<timestamp>/` containing the resolved config, the git SHA and
dirty flag, an environment snapshot (torch/transformers/CUDA versions, `nvidia-smi`),
`metrics.jsonl` and `summary.json`. The logging rules are in [research/README.md](research/README.md).

## From experiments to paper (ARS)

```text
research/ + experiments/E###/provenance.yaml + research/literature/papers/
      │  scripts/export_provenance.py     scripts/build_corpus.sh
      ▼
paper/passports/  ──►  /academic-research-skills:academic-pipeline  ──►  paper/stage1 … stage6
```

See [paper/README.md](paper/README.md) for the stage-to-folder map and the rules ARS enforces.

## Environment

The target stack is torch 2.8 (CUDA wheels), transformers 5.x (≥ 5.4) and VLMEvalKit, on NVIDIA
GPUs. Library code stays importable without a GPU, so the math can be checked on a laptop.

## Provenance

This repository is hosted at `ashutosh-mulc/cuekeep-code` (private). On 2026-11-15 it replaced
that repo's `main`, which was a fork of the TPRL codebase
(Cao et al., *Language-Guided Token Compression with RL in LVLMs*, arXiv 2603.13394). The fork is
kept on the `archive/tprl` branch at `6617257` and is not developed further. The few
modules carried over are listed in [third_party/README.md](third_party/README.md), and the reasons
are recorded in [research/decisions/0003-port-from-tprl.md](research/decisions/0003-port-from-tprl.md).
