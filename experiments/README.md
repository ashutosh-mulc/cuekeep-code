# Experiments

Each experiment is a folder `E###-slug/`, created with `python scripts/new_experiment.py "title"`:

| File | Role |
|---|---|
| `README.md` | **Pre-registered** question, hypothesis, success criterion, and setup, written *before* running. Then the results log and the conclusion |
| `config.yaml` | Extends `configs/…`; the only place knobs are set |
| `run.py` | `main(cfg, run)`: the experiment logic. Launch with `python scripts/run.py experiments/E###-…/config.yaml` |
| `provenance.yaml` | ARS `experiment_provenance[]` entry (planned vs executed, negative results, limitations). Kept in sync as runs land; exported to the paper's Material Passport |
| `results/` | Curated outputs worth keeping in git (tables, small JSON, plots). Raw runs stay in `runs/` (gitignored) |

Lifecycle: ☐ planned → ◐ running → ✓ supported / ✗ refuted / ~ inconclusive. Never delete a refuted
experiment; its `negative_results` feed the paper's limitations.

## Index

| ID | Title | Resolves | Status | Key result | Created |
|---|---|---|---|---|---|
| [E000](E000-env-smoke/) | Server environment smoke test + trace-length census | M1 setup | ☐ planned | — | 2026-11-11 |
| [E001](E001-pooled-attention-fidelity/) | Pooled stand-in fidelity (log m bias, covariance correction) | Q1, Q2 | ☐ planned | — | 2026-11-11 |
| [E002](E002-full-kv-reference/) | Full-KV reference on the thinking suite | M1 reference | ☐ planned | — | 2026-11-11 |
<!-- index:end -->
