# Experiments

The plan is staged: small **gates** on fixed dev-split samples decide go / no-go, and each **main**
tier is held until its gates pass. The design, thresholds and on-fail actions are in
[research/experiment_design.md](../research/experiment_design.md); the live status is in
[gates.yaml](gates.yaml) (`python scripts/gate.py list`).

Each experiment is a folder `E###-slug/`, created with `python scripts/new_experiment.py "title"`:

| File | Role |
|---|---|
| `README.md` | **Pre-registered** question, hypothesis, and per tier: sample, pass criteria, on-fail action, what it unlocks. Written *before* running. Then the runs, results, gate decision and conclusion |
| `config.<tier>.yaml` | One config per tier (`gate`, `pilot`, `main`, `synthetic`). Sets `tier`, `requires_gates`, the sample, and the gate thresholds |
| `run.py` | `main(cfg, run)`: the experiment logic. It writes the gate verdict into the summary as `gate/*` |
| `provenance.yaml` | ARS `experiment_provenance[]` entry (planned vs executed, negative results, limitations) |
| `results/` | Curated outputs worth keeping in git. Raw runs stay in `runs/` (gitignored) |

```bash
python scripts/run.py experiments/E003-attention-shift/config.gate.yaml dump_path=<dir>   # refuses if a required gate has not passed
python scripts/gate.py set G3 passed --evidence runs/E003-attention-shift/<ts>-gate --note "gap 0.14 on ScreenSpot-Pro"
```

Lifecycle: ☐ planned → ◐ running → ✓ supported / ✗ refuted / ~ inconclusive. Never delete a refuted
experiment or a failed gate; its `negative_results` feed the paper's limitations.

## Index

| ID | Title | Gate | Main tier held by | Status | Key result | Created |
|---|---|---|---|---|---|---|
| [E000](E000-env-smoke/) | Environment smoke test + premise census | G1 | — (gate only) | ☐ planned | — | 2026-11-11 |
| [E001](E001-pooled-attention-fidelity/) | Pooled stand-in fidelity | G2 | G2 | ☐ planned | — | 2026-11-11 |
| [E002](E002-full-kv-reference/) | Full-KV reference on the thinking suite | G6 | G6 + cost approval | ☐ planned | — | 2026-11-11 |
| [E003](E003-attention-shift/) | Attention shift during decoding | G3 | G3 | ☐ planned | — | 2026-11-12 |
| [E004](E004-teacher-forced-paging-fidelity/) | Paging fidelity under teacher forcing | G4 | — (gate only) | ☐ planned | — | 2026-11-12 |
| [E005](E005-paging-overhead-microbench/) | Paging overhead microbenchmark | G5 | — (gate only) | ☐ planned | — | 2026-11-12 |
| [E006](E006-accuracy-memory-tradeoff/) | Accuracy vs visual-KV memory | G7 (pilot) | G3, G4, G6, G7 | ☐ planned | — | 2026-11-12 |
<!-- index:end -->
