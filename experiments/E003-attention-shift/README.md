# E003: Attention shift during decoding

- **Created:** 2026-11-12 · **Status:** ☐ planned · **Resolves:** gate G3; hypothesis H2; Q3
- **Provenance:** [provenance.yaml](provenance.yaml)
- **Tiers:** gate G3 ([config.gate.yaml](config.gate.yaml), requires G1) → main ([config.main.yaml](config.main.yaml), requires G3).

## Pre-registration (written 2026-11-12, before any run)

**Question.** Does the set of image regions a thinking trace attends to change during decoding? If a
set chosen once serves the whole trace, recalling regions has nothing to add over pruning once.

**Hypothesis (H2).** It changes: an oracle that re-picks the top regions at every step covers clearly
more visual attention mass than a set fixed after the first steps. A set picked one step earlier
covers almost as much as the oracle, so paging with a 1-step lookahead can track the need.

**Method.** From the shared dump (true keys, true queries, text summary), compute each step's exact
region attention mass, summed over heads. At a budget of a fraction f of the regions, compare the
coverage of visual attention mass by:
- **static:** top regions over the first 5% of sampled steps, then fixed (the best case for pruning once);
- **lagged:** the previous sampled step's top regions;
- **dynamic:** the current step's top regions (oracle).

### Gate tier (G3)

- **Sample.** The gate dump (shared with E001): 8 ScreenSpot-Pro + 8 MathVista dev samples × 6 layers
  × 64 spread steps; plus dense windows of 256 consecutive steps for 2 + 2 samples.
- **Pass criteria** (computed into `summary.json` as `gate/*`):
  1. At f = 10% and g = 4, the median (dynamic − static) coverage over spread units is ≥ 0.10 on at
     least one dataset.
  2. On the dense windows, 1-step-lagged coverage is within 0.05 of dynamic.
- **On fail.** Gap < 0.05 on both datasets (`gate/kill`): (human). Recall has no edge over pruning
  once, and the project needs redirecting before any paging code is written. Gap between 0.05 and
  0.10 (`gate/marginal`): (human). Criterion 2 alone: paging needs a predictive score or a longer
  residency; (human).
- **Unlocks.** The main tier; gate G7; together with G2, the masked-simulation hook.

### Main tier (held by G3)

- **Requires gates.** G3.
- **Sample.** The main dump: 30 + 30 test samples × 13 layers.
- **Success criteria.** The gap holds on test samples; reported by layer, dataset and trace position
  (early / middle / late), with turnover and the share of late needs never in the static set. This
  is the paper's motivation figure.

## Runs

| Run dir | Tier | Date | Overrides | Notes |
|---|---|---|---|---|
| | | | | |

## Results

## Gate decision

## Conclusion
