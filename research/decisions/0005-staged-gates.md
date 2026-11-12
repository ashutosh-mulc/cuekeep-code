# 0005: Staged gates hold back the main experiments

- **Date:** 2026-11-12
- **Status:** accepted

## Context
The first plan went straight to full-size runs: a 100-sample, 13-layer KV dump of about 100 GB, and
the full benchmark suite with 3 seeds. Nothing cheap tested the assumptions those runs depend on.
The most important one, that the regions a trace needs shift during decoding (H2), had no experiment
at all before the main accuracy runs. The user asked for small gates on sampled data (not tiny, not
the full dataset) that the main experiments wait for.

## Options considered
1. **Keep full-size runs, add informal smoke tests.** Cheap to write, but nothing stops a main run
   from launching on an unverified assumption, and there are no recorded go / no-go decisions.
2. **Staged gates, enforced by the runner.** Each large run and each large piece of engineering
   waits for a small pre-registered gate. More files and a ledger to keep, but the order of work
   follows the order in which the idea could fail.

## Decision
Option 2. The design is in [experiment_design.md](../experiment_design.md):
- Eight gates, G0–G7, each with a sample, pass thresholds, an on-fail action and what it unlocks.
  `experiments/gates.yaml` is the ledger; `scripts/run.py` refuses a config whose `requires_gates`
  have not passed; overrides are recorded; only the human can waive a gate.
- Gates and all tuning use a fixed dev split (150 per dataset, nested subsets). Main tables report
  the full benchmark and the test split alone.
- The KV dump stores a two-tensor summary of the text keys per step instead of the text KV. The gate
  dump is about 3.5 GB at worst and the main dump about 27 GB at worst (was about 100 GB), and one
  capture serves two gates (stand-in fidelity and attention shift).
- Two new low-variance measures carry the evidence that paging beats pruning: coverage of attention
  mass (G3) and next-token KL under teacher forcing (G4). Free-running accuracy on 150 samples is
  too noisy to rank methods, so G6 and G7 only catch breakage and collapse.
- Engineering is gated too: the masked-simulation hook waits for G2 + G3, the real offload path for
  G4 + G5, and the remaining baselines for G7.

## Consequences
- New experiments E003 (attention shift), E004 (teacher-forced fidelity), E005 (overhead
  microbenchmark) and E006 (accuracy vs memory); E000–E002 were re-tiered. All pre-registrations were
  written or revised before any real run.
- The thresholds are first proposals made without data. Changing one after its gate has run needs
  the human and a note here.
- A failed G3 stops the project's current form early, after about one day of server work instead of
  after the paging system and the baselines are built.
