# Server checklist (GPU agent)

Rewritten 2026-11-15. Read this before doing anything on the server.

## How this works now

- **The server cannot push to git.** All code is written and tested on the laptop, pushed, and
  pulled here. The server agent **runs prepared commands only**. Do not edit tracked files: the edits
  cannot leave the server, and they make every run `git_dirty`.
- If something fails, do not patch the code here. Stop that step, and report the full error and log.
- **Report back** by encoding the run folders, without the large `.pt` files:
  `encode_dir.sh <runs/...> > out.txt` (it keeps `.json`, `.jsonl`, `.yaml` and `.log`). Paste it, with a
  short status: what ran, `gate/*` from each `summary.json`, and anything unexpected.
- **Do not record gate decisions here.** Report the computed `gate/*` values. The ledger
  (`experiments/gates.yaml`) is updated on the laptop.
- **The GPU is shared** with other jobs; that is normal. Timing numbers are noisy, so report medians.
- Background: [research/experiment_design.md](research/experiment_design.md). Gate status:
  `python scripts/gate.py list` (G0 ✓, G1 ✓, G5 ✗ round 0, others pending, as of 2026-11-15).

---

## Batch 1: the gate dump, then G3 and G2 (prepared 2026-11-15)

### 1. Sync

```bash
cd <repo>                                   # /group-volume/a.mulchandan/Research/ICML2027/cuekeep-code
git stash push -u -m "server-local-2026-11-15"   # keeps the earlier local edits; runs/ is untouched (ignored)
git pull
git log -1 --oneline                        # must be the commit named in the laptop's message
git status --short                          # must be empty
source .venv/bin/activate
```

### 2. Tests (about 2 minutes)

```bash
pytest -q                                   # expect: all passed, none failed (57 at the time of writing)
```

This includes:
- the CPU math tests;
- the loaders;
- an end-to-end run of the model code on a tiny random Qwen3-VL (it reads the real model's config and
  processor from the HF cache; no weights);
- three CUDA tests of the paging cache (`tests/gpu`), which exercise streams, events and pinned memory.

Any failure: stop and report.

### 3. One-minute CUDA dry run of the dump

```bash
python -m cuekeep.testing.tiny_qwen3vl /tmp/cuekeep_tiny
python scripts/dump_visual_kv.py configs/dump/tiny.yaml model.id=/tmp/cuekeep_tiny \
    device=cuda model.device_map=cuda:0 allow_cpu=false --tag cuda-dryrun
```

Expect `samples_ok: 3` and `units: 8`. This checks the capture hook on CUDA before the real run.

### 4. The gate dump (roughly 30–90 minutes; the GPU is shared)

```bash
GPUS=0 SCRIPT=scripts/dump_visual_kv.py bash scripts/launch.sh configs/dump/gate.yaml
```

- **Input:** 8 ScreenSpot-Pro (local release; GUI-click prompt) + 8 MathVista testmini, the first 8 of
  each dev split. Layers 2, 5, 11, 17, 24, 35; 64 spread steps; dense windows of 256 steps for 2 + 2 samples.
- **Output:** `runs/DUMP-gate/<ts>/`, containing `units/*.pt` (a few GB at most), `manifest.jsonl` (one
  row per sample: trace length, truncation, visual tokens, click score) and `summary.json`.
- **Check before continuing:** `samples_ok` is 16 (or say which samples were skipped and why), and the size is ≤ 5 GB.

### 5. G3: attention shift (minutes)

```bash
python scripts/run.py experiments/E003-attention-shift/config.gate.yaml dump_path=runs/DUMP-gate/<ts>/units
```

### 6. G2: stand-in fidelity (minutes to an hour)

```bash
python scripts/run.py experiments/E001-pooled-attention-fidelity/config.gate.yaml dump_path=runs/DUMP-gate/<ts>/units
```

If `gate/capture_max_rel_err` ≥ 0.01, the capture is suspect: report it, and do not trust the other numbers.

### 7. Report

Encode and paste these three folders: `runs/DUMP-gate`, `runs/E003-attention-shift`,
`runs/E001-pooled-attention-fidelity`, plus `runs/DUMP-tiny` from step 3. Include in the status:
- the dump summary: median trace length per dataset, truncation rate, click parse and hit rates;
- E003's `gate/*`;
- E001's `gate/*`.

Do not start anything else from the list below; it needs code that is not written yet.

---

## Later batches (code still to be prepared on the laptop)

| Gate | Experiment | Waits for | Status of the code |
|---|---|---|---|
| G5 round 1 | E005 | — | Optimisation (batched scoring across layers, persistent attention buffers) and an in-process baseline; the human decides how the baseline judges the gate |
| G6 | E002 | G1 ✓ | VLMEvalKit wrapper not written; `run.py` is a stub |
| G4 | E004 | G2, G3 | Masked-simulation hook not written |
| G7 | E006 | G3, G4, G6 | Not written |
| Main tiers | E001, E003 main dump | G2, G3 | Dump script ready (`configs/dump/` main config to be added) |
