# E004: Paging fidelity under teacher forcing

- **Created:** 2026-11-12 · **Status:** ☐ planned · **Resolves:** gate G4; Q10, Q12
- **Provenance:** [provenance.yaml](provenance.yaml) · **Tier:** gate only ([config.gate.yaml](config.gate.yaml)); requires G2 and G3
- Implemented 2026-11-16 ahead of its gates (decisions/0006): `src/cuekeep/sim/paged_sim.py` and `run.py`, tested on a tiny Qwen3-VL. The run stays held until G2 and G3 pass.

## Pre-registration (written 2026-11-12, before any run)

**Question.** With only a fraction of visual KV exact, does paging keep the model's next-token
predictions close to full KV, and closer than pruning once with the same budget?

**Hypothesis.** At a 25% budget, CueKeep's KL to full KV is at most half of prune-once's, and its
top-1 token agrees with full KV at ≥ 95% of positions. Oracle paging is better still; the difference
is the headroom for a better fetch score (Q10).

**Method: teacher forcing, masked simulation.**
- Generate one full-KV trace per sample (the model's own sampling, per-sample seed) and keep the
  first 4,096 generated tokens.
- Replay that trace through the model under each condition, and compare each position's next-token
  distribution with full KV's. No free-running generation, so there is no sampling noise and no
  divergence of traces: differences are caused by the method alone.
- Masked simulation: all KV stays on the GPU. Stand-ins are added as extra keys with a log m bias,
  and residency is applied through the additive attention mask, per position. This is exactly
  `pooling.attend_mixed`. It tests accuracy without the offload machinery.
- The slot pool runs causally over positions in each paged layer, with `lookahead: 1`.

**Conditions**, each at visual budgets of 10% and 25%, matched in *entries*: exact tokens plus pooled
entries for the stand-in conditions; exact tokens only for prune-once.

| Condition | Residency | Rest of the image |
|---|---|---|
| Full KV | everything | — |
| CueKeep v1 | score-based slot pool (`configs/method/cuekeep_v0.yaml`, g and budget from the sweep) | stand-ins |
| Oracle paging | top regions by exact attention share at each position | stand-ins |
| Prune-once | top tokens by attention at the end of the prompt, fixed | dropped |
| Random regions | random set, fixed | stand-ins |
| Stand-ins only | none | stand-ins |

**Clarifications (2026-11-16, before any run; decisions/0006).**
- The reference for every comparison is full KV computed through the simulation's own float32 path
  (`all_resident`), so differences between conditions come from paging alone. Criterion 1 compares
  that path with the untouched model, which checks the hook.
- CueKeep is simulated with the batched protocol of G5 round 1. After decode step t, step t's query
  scores the regions; fetches are visible from t + lookahead, and evictions from t + 1. At the end of
  prefill, all slots are filled from the last prompt query's scores (`prefill_fill`).
- Criteria 2 and 3 are read at the method's region size, g = 4.
- The decode positions compared are those that predict trace tokens 1 … n−1. The first trace token
  comes from prefill, which is full attention for every condition.

### Gate tier (G4)

- **Sample.** 16 ScreenSpot-Pro + 16 MathVista dev samples (the first 16 of each dev split, so they
  include the 8 + 8 of the gate dump). g ∈ {2, 4}.
- **Pass criteria** (compute into `summary.json` as `gate/*`):
  1. Equivalence: at a 100% budget, top-1 agreement ≥ 0.999 and mean KL ≤ 1e-3. The hook is correct.
  2. At 25%: CueKeep's median KL ≤ 0.5 × prune-once's on at least one dataset, and not higher on the other.
  3. At 25%: CueKeep's top-1 agreement ≥ 0.95 on both datasets.
- **Reported, not gating.** CueKeep vs oracle (Q10); CueKeep vs random regions (is the score doing
  anything?); stand-ins only (how much the stand-ins alone carry); KL by trace position.
- **On fail.** Criterion 1: fix the hook; nothing else counts until it holds. Criterion 2: (human);
  paging shows no fidelity edge over pruning. Criterion 3 alone: tune g and the budget on dev, then (human).
- **Unlocks.** Gate G7; with G5, the real offload build.

## Runs

| Run dir | Tier | Date | Overrides | Notes |
|---|---|---|---|---|
| | | | | |

## Results

## Gate decision

## Conclusion
