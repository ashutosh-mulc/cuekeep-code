# Experiment design: staged gates

_Last updated: 2026-11-12 · Decision record: [decisions/0005](decisions/0005-staged-gates.md) · Ledger: [experiments/gates.yaml](../experiments/gates.yaml)_

## Principles

1. **The cheapest test that could kill the idea runs first.** Every large run, and every large piece of
   engineering, sits behind a small gate that tests the assumption it depends on.
2. **Gates are small but not tiny.** They use a fixed seeded subset of the *dev split*: 16–40 samples
   where each sample is expensive (KV capture, teacher forcing), and 150 per benchmark where the
   measure is accuracy. They never use the full dataset.
3. **Gates are pre-registered go / no-go decisions.** Each has pass thresholds, an on-fail action,
   and a list of what it unlocks. A failed gate is a result: record it, then follow the on-fail action.
4. **Gates are enforced.** A config lists `requires_gates`, and `scripts/run.py` refuses to start it
   until the ledger shows them passed. Overriding needs `--override-gates REASON`, which is
   recorded in the run. Only the human can waive a gate.
5. **Gate numbers never go in the paper.** They decide what to run. Paper numbers come from main tiers.
6. **Nothing is tuned on test data.** Gates and every design choice (g, budget, score rule) use the
   dev split. Main tables report the full benchmark (comparable with published numbers) and the test
   split alone (untouched by tuning).

## Gates

Thresholds are first proposals, set before any data exists. Changing one after its gate has run
needs the human and an entry in the decision log.

| Gate | Tests | Experiment and sample | Pass | On fail | Unlocks |
|---|---|---|---|---|---|
| **G0** Environment | The stack works; the GPU-only code paths are correct | No dataset. `env_check.py --model`; `pytest -m gpu` (CUDA port of the cache tests); the timing hook on one sample | Expected model shapes; bf16 and float-mask SDPA work; GPU tests pass; prefill/decode split recorded | Fix and rerun. Nothing else starts | G1, G5 |
| **G1** Premise | Traces are long, visual KV is a real share, tokens are row-major | [E000](../experiments/E000-env-smoke/): 20 MME + 20 ScreenSpot-Pro (dev), 1 seed | Runs finish; `token_order_row_major` true; median trace ≥ 300 new tokens on ≥ 1 dataset; on ScreenSpot-Pro, visual tokens ≥ 50% of prompt tokens (median) | Token order false: stop, (human). Short traces or few visual tokens: (human), since the premise is weak for this model or data | G2, G3, G5, G6 |
| **G2** Stand-in fidelity | Pooled stand-ins + log m work on real keys | [E001](../experiments/E001-pooled-attention-fidelity/) gate tier, on the **gate dump** (below) | (1) capture check < 1e-2; (2) the log m bias lowers all-pooled error in ≥ 80% of units at every g; (3) with the oracle top 25% of regions exact, median output error ≤ 0.10 at g ∈ {2, 4} in ≥ 75% of units, and stand-ins beat dropping the rest in ≥ 75% of units | (3) fails: rerun on pre-RoPE keys from the same dump, then (human). (2) fails: (human); a claimed contribution is at risk | E001 main, G4 |
| **G3** Attention shift | The regions a trace needs change during decoding (H2) | [E003](../experiments/E003-attention-shift/) gate tier, same gate dump | At a 10% region budget and g = 4, median (dynamic − static) coverage ≥ 0.10 on ≥ 1 dataset; on the dense windows, 1-step-lagged coverage within 0.05 of dynamic | Gap < 0.05 on both datasets: (human). Recall then has no edge over pruning once, and the project needs redirecting. Gap 0.05–0.10: (human) | E003 main, G7 |
| **G4** Paging fidelity | Paging keeps next-token predictions close to full KV, and closer than pruning | [E004](../experiments/E004-teacher-forced-paging-fidelity/): 16 ScreenSpot-Pro + 16 MathVista (dev), full-KV trace replayed (first 4,096 tokens), masked simulation | (1) at 100% budget: top-1 agreement ≥ 0.999 and KL ≤ 1e-3; (2) at 25% budget: CueKeep's median KL ≤ 0.5 × prune-once KL on ≥ 1 dataset and not worse on the other; (3) CueKeep top-1 agreement ≥ 0.95 at 25% | (1): the hook is wrong; fix it. (2): (human), no fidelity edge over pruning. (3) alone: tune g and budget on dev, then (human) | G7, real offload build |
| **G5** Overhead | Scoring, commit and fetch are cheap next to a decode step | [E005](../experiments/E005-paging-overhead-microbench/): no dataset; real model shapes; G1's decode ms/token | Per-step paging overhead ≤ 15% of the full-KV decode step at `refresh_every` ≤ 4; commit wait ≤ 5% of the step with `lookahead: 1` | Optimise (batch scoring across layers, remove the per-layer host sync, share decisions across layers), then rerun | Real offload build |
| **G6** Reference sanity | Our harness reproduces the model | [E002](../experiments/E002-full-kv-reference/) gate tier: 150 per benchmark (dev), 1 seed | Every benchmark runs; none more than 8 points below its official score; truncation < 5%; GPU-hour estimate for the main tier written down | Fix prompting or answer extraction; rerun | E002 main, G7 |
| **G7** Accuracy pilot | Free-running accuracy does not collapse under paging | [E006](../experiments/E006-accuracy-memory-tradeoff/) pilot tier: 150 × 2 benchmarks (dev), 1 seed; full KV, CueKeep, prune-once | CueKeep at 25% budget within 4 points of full KV on both benchmarks, and no more than 2 points below prune-once at the matched budget | (human). Check trace lengths first (longer traces can hide or cause the drop) | E006 main, baseline implementations |

At n = 150, one accuracy number has a standard error of about 4 points. G6 and G7 therefore catch
breakage and collapse; they do not rank methods. The evidence that paging is *better* than pruning
comes from G3 and G4, which use low-variance measures: attention mass, and KL under teacher forcing.

## Order and dependencies

```
G0 ──► G1 ──┬──► [gate dump: one capture] ──┬──► G2 ──► G4 ──┐
            │                               └──► G3 ─────────┼──► G7 ──► E006 main + baselines
            ├──► G6 ─────────────────────────────────────────┘
            └──► G5 ──► (with G4) real offload build
```

- G2 and G3 read the same dump and take minutes of analysis each. Run both before deciding anything.
- G5 and G6 are independent of the dump and can run in parallel with it.
- The kill order is G1 → G3 → G2 → G4. If G3 fails, stop before writing the paging hook.

## Engineering that is also held

| Work | Held until | Why |
|---|---|---|
| Masked-simulation attention hook (stand-ins and residency through masks, all KV on the GPU) | G2 and G3 passed | It is only needed if stand-ins work and attention shifts |
| Real offload (pinned host store, async fetch, physical deletion from the HF cache) | G4 and G5 passed | Building the systems path is only worthwhile once paging is accurate in simulation and cheap per step |
| Baselines (FastV, PDrop, Quest, ShadowKV-style, VisionPulse-style, DSTP-style, SnapKV) | G7 passed | Prune-once is needed for G4 and G7; the rest are needed only for the main tables |
| VLMEvalKit wrapper | G1 passed | G6 needs it |

## Main experiments

| Experiment | Requires | Sample | Purpose |
|---|---|---|---|
| E001 main | G2 | **Main dump** | Paper-grade fidelity numbers by layer; pre- vs post-RoPE; score variants (Q1, Q2, Q10, Q12) |
| E003 main | G3 | Main dump | The attention-shift figure (H2) |
| E002 main | G6, cost approved by the human | Full suite, seeds {0, 1, 2} | Full-KV reference row of every table; seed variance (Q11) |
| E006 main | G3, G4, G6, G7 | Full suite, seeds {0, 1, 2}, all baselines, budget sweep | Accuracy vs visual-KV bytes; total response time; with real offload once G5 has passed |

## Dev and test split

- `cuekeep.data.split_indices` partitions each dataset's indices with a fixed seed (`DEV_SEED`).
  **Dev** is the first 150 indices of that permutation; **test** is the rest.
- Smaller gate samples take a prefix of dev, so they are nested: the 8 samples of a dump gate are
  inside the 20 of G1, which are inside the 150 of G6.
- For VLMEvalKit benchmarks, apply the same function to the benchmark's row order in the wrapper.
- Main-tier dumps sample from **test**.

## The shared dump

One capture serves G2 and G3 (and their main tiers). The format is in
`src/cuekeep/analysis/dumps.py`. Visual KV is stored once per (sample, layer). Each sampled decode
step stores the query, plus a two-tensor summary of the text keys' contribution (`other_lse`,
`other_out`) in place of the text KV. That keeps the dump at about 4 KB per visual token per
(sample, layer), plus about 16 KB per step.

| | Samples | Layers | Steps per sample | Worst-case size |
|---|---|---|---|---|
| **Gate dump** | 8 ScreenSpot-Pro + 8 MathVista (dev) | 2, 5, 11, 17, 24, 35 | 64 spread over the trace; plus, for 2 + 2 samples, a dense window of 256 consecutive steps starting 25% into the trace | about 3.5 GB |
| **Main dump** | 30 + 30 (test) | 0, 4, 5, 8, 11, 12, 16, 17, 20, 24, 28, 32, 35 | 64 spread; a dense window for 5 + 5 samples | about 27 GB |

Worst case assumes 8,000 visual tokens per image. The main dump is only captured after G2 and G3 pass.

Implementation (2026-11-15): `scripts/dump_visual_kv.py` with `configs/dump/gate.yaml`. Traces come
from ordinary generation; the capture is one teacher-forced pass with a temporary wrapper on the SDPA
attention function (`src/cuekeep/analysis/capture.py`). Pre-RoPE keys are not captured in this version.
ScreenSpot-Pro uses the benchmark's GUI-click prompt.

## Timing on a shared GPU (added 2026-11-15)

The server GPU is shared with other jobs, which is normal and cannot be avoided. E000 saw decode
times swing between 26 and 101 ms/token. So:
- Report medians, not means, for every timing.
- Never compare a timing with one measured in a different run. A timing gate measures its own
  baseline in the same process, interleaved with the treatment, so both see the same contention.
- For G5, this changes how the gate is judged after round 0 failed, so it needs the human's sign-off
  (see E005). Until then, round 1 reports both the pre-registered baseline and the in-process one.

## Cost bookkeeping

G1 measures decode speed r (tokens/s) and the median trace length T. Before launching any tier,
write its estimate in the notebook: samples × seeds × T / r, plus prefill. Any single launch
estimated above 24 GPU-hours needs the human's approval.
