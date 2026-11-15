# Server checklist (GPU agent)

Rewritten 2026-11-16 for the next server session. Read this before doing anything on the server.

## How this works

- **The server cannot push to git.** All code is written and tested on the laptop, pushed, and
  pulled here. The server agent **runs prepared commands only**. Do not edit tracked files: the edits
  cannot leave the server, and they make every run `git_dirty`.
- If something fails, do not patch the code here. Stop that step, report the full error and log, and
  carry on with the steps that do not depend on it.
- **Report back** by encoding the run folders, without the large `.pt` files:
  `encode_dir.sh <runs/...> > out.txt` (it keeps `.json`, `.jsonl`, `.yaml` and `.log`). Paste it with a
  short status: what ran, `gate/*` from each `summary.json`, and anything unexpected.
- **Gate decisions are recorded on the laptop**, not here. The ledger here stays as pulled.
- **The GPU is shared** with other jobs; that is normal. Report medians for timings.
- Background: [research/experiment_design.md](research/experiment_design.md). Status as of 2026-11-16:
  G0 ✓, G1 ✓, G5 ✗ (round 0); the others are pending.

---

## Step 0: sync, tests, data (about 15 minutes)

```bash
cd <repo>                                        # /group-volume/a.mulchandan/Research/ICML2027/cuekeep-code
git stash push -u -m "server-local-2026-11-15"   # once; keeps the earlier local edits. runs/ is ignored, so untouched
git pull
git log -1 --oneline                             # must be the commit named in the laptop's message
git status --short                               # must be empty
source .venv/bin/activate
pytest -q                                        # expect: 74 passed, none failed
python -c "from datasets import load_dataset; print(len(load_dataset('AI4Math/MathVista', split='testmini')))"   # expect 1000
```

`pytest -q` includes:
- the math, loader and scoring tests;
- the tiny-Qwen3-VL integration tests (the dump, E001, E003, E004 and E005 on CPU);
- three CUDA paging-cache tests.

Any failure: stop and report. The MathVista line downloads the dataset once (needed by the dump and
G6), so neither stalls later.

One-minute CUDA dry runs of the new code paths:

```bash
python -m cuekeep.testing.tiny_qwen3vl /tmp/cuekeep_tiny
python scripts/dump_visual_kv.py configs/dump/tiny.yaml model.id=/tmp/cuekeep_tiny device=cuda \
    model.device_map=cuda:0 allow_cpu=false --tag cuda-dryrun                    # expect samples_ok 3, units 8
python scripts/run.py experiments/E004-teacher-forced-paging-fidelity/config.gate.yaml model.id=/tmp/cuekeep_tiny \
    model.max_new_tokens=48 'method.layers_paged="1:"' 'datasets=[{name: synthetic, n: 1}]' \
    --tag cuda-dryrun --override-gates "tiny-model CUDA dry run"                 # expect gate/c1_equivalence true
```

## Step 1: G5 round 1, alone on our side (about 10 minutes)

Timing-sensitive, so start nothing else of ours while it runs:

```bash
python scripts/run.py experiments/E005-paging-overhead-microbench/config.gate.yaml --tag round1
```

Report the `gate/*` block, both the `gate/preregistered/*` and the `gate/in_process/*` values.
`gate/pass` follows the pre-registered baseline until the human decides otherwise.

## Step 2: the gate dump and G6, side by side

```bash
GPUS=0 SCRIPT=scripts/dump_visual_kv.py bash scripts/launch.sh configs/dump/gate.yaml     # about 1 hour
GPUS=0 bash scripts/launch.sh experiments/E002-full-kv-reference/config.gate.yaml          # 2-4 hours (300 generations)
```

(Use a second GPU for E002 if one is free: `GPUS=1`.)

- **Dump checks:** `samples_ok` is 16 (or say which samples were skipped and why), and the size is ≤ 5 GB.
- **G6 output:** `runs/E002-full-kv-reference/<ts>/samples.jsonl` has every answer with its score. If an
  accuracy is far below the official score, look at ~10 answers marked wrong and say whether they are
  really wrong or the extraction missed them.

## Step 3: G3 and G2 on the dump (minutes)

```bash
python scripts/run.py experiments/E003-attention-shift/config.gate.yaml dump_path=runs/DUMP-gate/<ts>/units
python scripts/run.py experiments/E001-pooled-attention-fidelity/config.gate.yaml dump_path=runs/DUMP-gate/<ts>/units
```

If E001's `gate/capture_max_rel_err` ≥ 0.01, the capture is suspect: report it, do not trust its
other numbers, and do not go on to step 4.

## Step 4: G4, only if G3 *and* G2 computed `gate/pass: true` (2-3 hours)

The ledger on the server still shows G2 and G3 as pending, so name the evidence in the override:

```bash
GPUS=0 bash scripts/launch.sh experiments/E004-teacher-forced-paging-fidelity/config.gate.yaml \
    --override-gates "G3 pass in runs/E003-attention-shift/<ts>, G2 pass in runs/E001-pooled-attention-fidelity/<ts>; ledger is updated on the laptop"
```

If either gate failed, or `gate/kill` is true in E003: do not run step 4. Report, and the human decides.

## Step 5 (optional, if steps 3-4 went well and time remains): the main dump

```bash
GPUS=0 SCRIPT=scripts/dump_visual_kv.py bash scripts/launch.sh configs/dump/main.yaml \
    --override-gates "G2 and G3 computed pass (runs named above)"
```

60 test-split samples, 13 layers; up to about 27 GB.

## Step 6: report

Encode and paste: `runs/E005-paging-overhead-microbench`, `runs/DUMP-gate` (and `runs/DUMP-main` if
run), `runs/E003-attention-shift`, `runs/E001-pooled-attention-fidelity`, `runs/E002-full-kv-reference`,
`runs/E004-teacher-forced-paging-fidelity`, `runs/DUMP-tiny`. Include in the status:
- every gate's `gate/*` block;
- the dump summary (median trace length per dataset, truncation rate, click parse and hit rates);
- G6 accuracies against the official scores;
- anything that failed, with its log.

---

## Not ready yet

| Gate | Experiment | Waits for | Code |
|---|---|---|---|
| G7 | E006 | G3, G4, G6 | Free-running paged generation not written; `run.py` is a stub |
| Main tiers | E001, E003 (main dump), E002 main | G2, G3, G6 + cost approval | Ready; launch only when the human says |
