# E006: Accuracy vs visual-KV memory

- **Created:** 2026-11-12 · **Status:** ☐ planned · **Resolves:** gate G7; hypotheses H2, H3; Q5
- **Provenance:** [provenance.yaml](provenance.yaml)
- **Tiers:** pilot G7 ([config.pilot.yaml](config.pilot.yaml), requires G3, G4, G6) → main ([config.main.yaml](config.main.yaml), requires G7).
- `run.py` is a stub. This is the paper's main experiment, and it is held by every earlier gate.

## Pre-registration (written 2026-11-12, before any run)

**Question.** In free-running generation, how much accuracy does CueKeep keep at a given visual-KV
budget, compared with full KV and with baselines at the same budget? What happens to trace length
and total response time?

**Hypothesis.** At 25% of visual KV, CueKeep is within 2 points of full KV on the suite and above
prune-once methods at the same budget, with the largest gap on high-resolution benchmarks. Traces
do not get longer (Q5).

**Setup (both tiers).** VLMEvalKit harness from E002; the model's own sampling; per-sample seeds
shared across methods; `max_new_tokens` 40960. Budgets are matched in visual-KV bytes on the GPU.
Every table reports accuracy, trace length, total response time and visual-KV bytes.

### Pilot tier (G7)

- **Sample.** 150 dev samples of each of 2 benchmarks (one reasoning-heavy, one high-resolution;
  choose them from E002's gate run), 1 seed. Methods: full KV, CueKeep, prune-once. Budget 25%.
  It may run on the masked simulation; real offload is not required.
- **Pass criteria.**
  1. CueKeep is within 4 points of full KV on both benchmarks.
  2. CueKeep is no more than 2 points below prune-once on either benchmark.
  3. CueKeep's median trace is no more than 15% longer than full KV's.
  At n = 150 one accuracy number has a standard error of about 4 points. This pilot catches a
  collapse; it does not rank methods. The fidelity evidence is G4.
- **On fail.** (human). Look at trace lengths first, then at per-sample disagreements with G4.
- **Unlocks.** The main tier, and the work of implementing the remaining baselines.

### Main tier (held by G7)

- **Requires gates.** G3, G4, G6, G7. Systems numbers (decode ms/token, PCIe bytes per token) also
  need the real offload path, which is held by G5.
- **Sample.** The full suite, base seeds {0, 1, 2}; budgets 10%, 25%, 50%; g chosen on dev.
  Methods: full KV, CueKeep, FastV, PDrop, Quest-over-regions, ShadowKV-style, VisionPulse-style,
  DSTP-style, SnapKV, random regions.
- **Success criteria.** The accuracy vs visual-KV curve; CueKeep on or above every baseline's curve
  on the high-resolution benchmarks, by more than the across-seed std; results on the full set and on
  the test split alone.

## Runs

| Run dir | Tier | Date | Overrides | Notes |
|---|---|---|---|---|
| | | | | |

## Results

## Gate decision

## Conclusion
