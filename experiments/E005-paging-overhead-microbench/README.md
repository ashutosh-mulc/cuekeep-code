# E005: Paging overhead microbenchmark

- **Created:** 2026-11-12 · **Status:** ☐ planned · **Resolves:** gate G5; Q4
- **Provenance:** [provenance.yaml](provenance.yaml) · **Tier:** gate only ([config.gate.yaml](config.gate.yaml)); requires G1

## Pre-registration (written 2026-11-12, before any run)

**Question.** Is the per-step cost of paging (score regions, plan, issue fetches, commit, build
attention inputs, across all 34 paged layers) small next to a full-KV decode step? Does a 1-step
lookahead hide the copy from host to GPU?

**Hypothesis.** The Python-level loop over layers and the per-layer host sync in `SlotPool.plan` are
the main costs. They fit in the budget at a refresh interval of 2–4 steps, if not at every step.

**Method.** No dataset and no model. Random K/V with the real shapes (8 KV heads, 32 heads, head dim
128, 34 layers), one `PagedVisualCache` per layer, random queries each step (worst-case churn).

### Gate tier (G5)

- **Sample.** 1k / 4k / 8k visual tokens; g ∈ {2, 4}; refresh interval ∈ {1, 2, 4}; 200 timed steps.
  Run on the GPU, with `bench.baseline_decode_ms_per_token` set from E000.
- **Pass criteria** (`gate/*`, at 4k visual tokens, g = 4, the best refresh interval ≤ 4):
  1. Paging overhead ≤ 15% of the full-KV decode step.
  2. Commit wait ≤ 5% of the decode step.
- **On fail.** Optimise, then rerun: score all layers in one batched call; drop the per-layer
  `.cpu()` in `SlotPool.plan`; share one paging decision across layers. This is engineering, so no
  (human) is needed unless two rounds fail.
- **Unlocks.** The real offload build (together with G4).

A CPU run only checks that the benchmark executes (`gate/valid` is false).

## Runs

| Run dir | Tier | Date | Overrides | Notes |
|---|---|---|---|---|
| | | | | |

## Results

## Gate decision

## Conclusion
