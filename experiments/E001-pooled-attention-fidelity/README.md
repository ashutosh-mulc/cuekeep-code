# E001: Pooled stand-in fidelity (log m bias, covariance correction)

- **Created:** 2026-11-11 · **Status:** ☐ planned · **Resolves:** gate G2; Q1, Q2, Q10, Q12 ([open_questions](../../research/open_questions.md))
- **Provenance:** [provenance.yaml](provenance.yaml)
- **Tiers:** synthetic code check ([config.synthetic.yaml](config.synthetic.yaml)) → gate G2
  ([config.gate.yaml](config.gate.yaml), requires G1) → main ([config.main.yaml](config.main.yaml), requires G2).
- Pre-registration revised 2026-11-12 before any real-key run: after the code review
  ([decisions/0004](../../research/decisions/0004-review-fixes-scoring-memory-fetch.md)) and for the staged-gate
  design ([decisions/0005](../../research/decisions/0005-staged-gates.md)). Only synthetic smoke runs exist.

## Pre-registration

**Question.** Do pooled stand-ins (mean K/V plus a log m bias) reproduce attention over image
regions well enough that keeping only a fraction of regions exact is accurate? Which score should
decide what to fetch?

**Hypotheses.**
- H1a: the log m bias reduces attention-output error relative to pooling without it, at every g.
- H1b: with the top 25% of regions exact and the rest pooled, output error is small (≤ 0.10), and
  smaller than when the rest are dropped.
- H1c: the *capped* covariance-corrected score ranks regions by true mass better than the plain
  pooled logit. The uncapped quadratic score does not, at realistic logit scales.
- H1d: per-head share normalisation predicts the head-averaged benefit of fetching better than a raw
  max over heads (Q12).
- H1e (exploratory, Q10): mass-based scores are only weak predictors of the oracle fetch benefit. We
  record how weak, as the headroom for a better priority rule.

### Synthetic tier (code check)

CPU, a smooth random key/value field over a ragged 23×30 grid with 256 text tokens, logit std ≈ 1–11.
It checks that the runner and metrics execute. Its numbers are not results and decide nothing.

### Gate tier (G2)

- **Sample.** The gate dump: 8 ScreenSpot-Pro + 8 MathVista dev samples × layers {2, 5, 11, 17, 24, 35}
  × 64 decode steps (format: `src/cuekeep/analysis/dumps.py`). Sweep g ∈ {2, 4, 8}.
- **Pass criteria** (computed into `summary.json` as `gate/*`, on post-RoPE units):
  1. Capture check: our recomputed attention matches the model's recorded output, relative error < 0.01.
  2. The log m bias lowers the all-pooled median error in ≥ 80% of units, at every g.
  3. With the oracle top 25% of regions exact per step and the rest pooled, median output error is
     ≤ 0.10 at g ∈ {2, 4} in ≥ 75% of units; and stand-ins beat dropping the rest in ≥ 75% of units.
- **On fail.** Criterion 1: the capture is wrong; fix `dump_visual_kv.py`. Criterion 3: rerun on the
  pre-RoPE keys in the same dump (cheap), then (human). Criterion 2: (human); the bias is a claimed
  contribution.
- **Unlocks.** The main tier (and the main dump), and gate G4.

### Main tier (held by G2)

- **Requires gates.** G2.
- **Sample.** The main dump: 30 + 30 *test* samples × 13 layers × 64 steps, post- and pre-RoPE.
- **Success criteria.**
  1. Criteria 2 and 3 of the gate hold on the test samples, reported per layer.
  2. Top-8 recall of the true highest-mass regions improves by ≥ 5 points for the capped score at the
     best r vs r = 0. If not, `cov_rank` stays 0 in the method.
  3. Per-head share aggregation beats raw max in top-8 recall of the head-averaged oracle benefit,
     in ≥ 60% of layers.

**Planned measurements.** Mirrored in `provenance.yaml`: output error vs g by layer; the effect of
the log m bias; error at a 10% and 25% budget, stand-ins vs dropping; score variants vs exact region
mass and vs oracle fetch benefit; cross-head aggregation rules; pre- vs post-RoPE pooling.

## Runs

| Run dir | Tier | Date | Overrides | Notes |
|---|---|---|---|---|
| | | | | |

## Results

## Gate decision

## Conclusion
