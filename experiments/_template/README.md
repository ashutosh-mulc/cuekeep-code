# {{ID}}: {{TITLE}}

- **Created:** {{DATE}} · **Status:** ☐ planned · **Resolves:** {{QUESTION}}
- **Owner:** · **Provenance:** [provenance.yaml](provenance.yaml)
- **Tiers:** gate ([config.gate.yaml](config.gate.yaml)) → main ([config.main.yaml](config.main.yaml)).
  The main tier is held until this experiment's gate is passed in [../gates.yaml](../gates.yaml).

## Pre-registration (write before running; do not edit afterwards)

**Question.** What do we want to know?

**Hypothesis.** What we expect, and why.

### Gate tier (small, decides go / no-go)

- **Sample.** A fixed seeded subset of the dev split: how many samples, layers, steps, seeds.
- **Pass criteria.** Concrete thresholds, each checkable from the run's `summary.json`.
- **On fail.** What happens next: the cheap rescue to try, then who decides (human).
- **Unlocks.** Which runs or engineering work this gate holds back.

### Main tier (held by the gate)

- **Requires gates.** IDs from `../gates.yaml` (also set `requires_gates` in `config.main.yaml`).
- **Sample.** Full size: datasets, seeds.
- **Success criteria.** What the paper claim needs.

**Planned measurements.** Mirror these in `provenance.yaml → planned_vs_executed`.

## Runs

| Run dir | Tier | Date | Overrides | Notes |
|---|---|---|---|---|
| | | | | |

## Results

## Gate decision

Passed / failed / waived, with the numbers against each pass criterion. Record it with
`python scripts/gate.py set <G#> <status> --evidence <run dir> --note "..."`.

## Conclusion

Supported / refuted / inconclusive, and what changes as a result (method, roadmap, open questions).
