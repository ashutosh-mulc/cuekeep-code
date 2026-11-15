# E005: Paging overhead microbenchmark

- **Created:** 2026-11-12 · **Status:** ✗ gate G5 failed, round 0 (2026-11-15); optimisation round 1 pending · **Resolves:** gate G5; Q4
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
| `runs/E005-paging-overhead-microbench/20261116-000009-gate` | gate | 2026-11-15 | `bench.baseline_decode_ms_per_token=42.9` | Round 0. Commit `1053902`, `git_dirty: true` (server-side changes not in the repo). Shared H100 (23.5 GB held by other processes at start) |

Curated copies: [results/](results/).

## Results

Per decode step, summed over 34 paged layers (ms); random queries each step (worst-case churn).

| Visual tokens | g | refresh_every | Regions | Total | Score | Commit | Inputs | Total / 42.9 ms |
|---|---|---|---|---|---|---|---|---|
| 4,096 | 4 | 1 | 256 | 89.1 | 87.4 | 2.70 | 3.8 | 2.08 |
| 4,096 | 4 | 2 | 256 | 47.6 | 44.8 | 1.38 | 3.8 | 1.11 |
| **4,096** | **4** | **4** | **256** | **25.7** | **23.7** | **0.71** | **3.9** | **0.60** |
| 4,096 | 2 | 4 | 1,024 | 26.9 | 25.7 | 0.69 | 3.9 | 0.63 |
| 8,100 | 2 | 1 | 2,025 | 169.9 | 174.6 | 2.68 | 4.5 | 3.96 |
| 8,100 | 4 | 4 | 529 | 25.9 | 23.6 | 0.70 | 4.2 | 0.60 |

(All 18 settings in `results/metrics_20261116-000009-gate.jsonl`.)

- **The cost is fixed per layer and per refresh, not proportional to work.** At g = 4, scoring costs
  about 87 ms per refresh whether there are 64 or 529 regions (about 2.6 ms per layer), while copies
  are only about 4 MB per step. It is host synchronisation (`.cpu()` in `SlotPool.plan`), Python
  bookkeeping and small kernel launches, repeated for 34 layers. At g = 2 it also grows with the
  region count (8,100 tokens: 2,025 regions).
- **Building attention inputs costs about 4 ms per step, even with no refresh**: `attention_inputs`
  concatenates the pooled table and the slot buffers for every layer, every step. That alone is about
  9–13% of a decode step.
- **The lookahead works** (Q4): waiting for fetches (`commit`) adds 0.7 ms per step, 1.7% of the step.
- Baseline: the pre-registered baseline is E000's *mean* decode ms/token (42.9), which GPU-sharing
  spikes inflate. Against E000's median (30.2) the best point is 0.85×.

## Gate decision

**G5 failed, round 0** (recorded 2026-11-15, evidence `runs/E005-paging-overhead-microbench/20261116-000009-gate`):
1. Overhead ≤ 0.15 of the decode step at 4,096 tokens, g = 4, best refresh_every ≤ 4: **0.60 ✗**
2. Commit wait ≤ 0.05: **0.017 ✓**

On-fail action (pre-registered): optimise, then rerun. Round 1 will (a) score all paged layers in one
batched call with one host sync per refresh, (b) keep persistent K / V / bias buffers so building
attention inputs returns views, (c) profile one step first. Sharing one paging decision across layers
changes the method that G4 evaluates, so it needs the human.
**Open decision (human):** with a shared GPU, a baseline measured in another run is unreliable. The
proposal for round 1 is to measure the full-KV decode step inside the same E005 process, interleaved
with the paging measurements, and to compare medians. Until decided, round 1 reports both that and
the pre-registered 42.9 ms.

## Round 1 (prepared 2026-11-16, not yet run)

Changes (decisions/0006):
- persistent device buffers, so attention inputs are views;
- scoring batched across all paged layers, with one host sync per refresh;
- `SlotPool.plan` on plain Python lists (about 6× faster per call);
- both scoring paths are measured (`batched`, `per_layer`).

**In-process baseline.** The real model decodes at 4,096 visual tokens in 8 rounds of 32 tokens,
interleaved with paging steps at the gate point. Medians are compared.

**Judging.** Both verdicts are reported. `gate/pass` follows `gate.judge: preregistered` (E000's mean,
42.9 ms) until the human decides whether the in-process baseline should judge.

Laptop check (MPS, 34 layers, 4,096 tokens, g = 4; not representative of CUDA copies):
| | refresh_every 1 | refresh_every 4 |
|---|---|---|
| Round 0 code | 42.3 ms/step | 15.9 ms/step |
| Round 1, batched | 28.1 ms/step | 6.8 ms/step |

## Conclusion

Paging bookkeeping in its first implementation is about 4× over budget, for engineering reasons
(per-layer host syncs, Python loops, per-step concatenation), not because of PCIe bandwidth or
algorithmic cost. The 1-step lookahead hides the copy itself. G5 holds back only the real offload
build, which also waits for G4, so round 1 comes after the gate dump (G3, G2).
