# E000: Environment smoke test + premise census (gate G1)

- **Created:** 2026-11-11 · **Status:** ✓ gate G1 passed (2026-11-15), with a caveat on the ScreenSpot-Pro prompt · **Resolves:** gate G1, M1 setup
- **Provenance:** [provenance.yaml](provenance.yaml) · **Tier:** gate only ([config.gate.yaml](config.gate.yaml)); requires G0
- Pre-registration revised 2026-11-12, before any run: once after the code review, once for the
  staged-gate design ([experiment_design](../../research/experiment_design.md)).

## Pre-registration

**Question.** Does Qwen3-VL-4B-Thinking load and generate on the server with the pinned stack? Is
the premise of the project true for this model: are thinking traces long, and is visual KV a real
share of what decoding holds?

**Hypothesis.** Traces are 1k–8k tokens. On high-resolution inputs, visual tokens are most of the
prompt. Per request, visual KV is still a minority of decode-time KV for long traces, and far smaller
than the weights (about 147 KB/token: 36 layers × 8 KV heads × 128 × 2 × bf16).

### Gate tier (G1)

- **Sample.** The first 20 samples of the dev split of MME and of ScreenSpot-Pro, 1 seed. 1 GPU,
  bf16, SDPA, the model's own sampling settings, `max_new_tokens=16384`.
- **Pass criteria** (computed into `summary.json` as `gate/*`):
  1. `token_order_row_major` is true.
  2. Median trace ≥ 300 new tokens on at least one dataset.
  3. On ScreenSpot-Pro, the median of visual tokens / prompt tokens is ≥ 0.50.
  4. No truncated sample. If any truncates, rerun with `model.max_new_tokens=40960` and judge that run.
- **Also required, by hand:** the ScreenSpot-Pro source, field names and box format are pinned in the
  config and checked by drawing 3 boxes.
- **On fail.** Criterion 1: stop, (human); region tiling would be wrong. Criteria 2 or 3: (human);
  the premise is weak for this model or data (options: higher-resolution data, the 8B model).
- **Unlocks.** G2, G3 (the gate dump), G5 (needs this run's decode ms/token), G6.

**Planned measurements.** Per sample: visual tokens, merged grid, trace length, truncation flag,
prefill seconds, decode ms/token, prefill and decode peak memory, visual and total KV bytes.

## Runs

| Run dir | Tier | Date | Overrides | Notes |
|---|---|---|---|---|
| `runs/E000-env-smoke/20261115-202408-gate` | gate | 2026-11-15 | — | Aborted: ScreenSpot-Pro from `Voxel51/ScreenSpot-Pro` downloaded one file per ~42 s (1,582 files, unauthenticated). No summary |
| `runs/E000-env-smoke/20261115-234047-gate` | gate | 2026-11-15 | — | **The gate run.** ScreenSpot-Pro from the original release on local disk (`source_annotations: local`). Commit `1053902` with uncommitted server-side changes (`git_dirty: true`): the local-annotation loader and a rewritten token-order check, neither of which reached the repo. The repo's loader (2026-11-15) reproduces the same sample indices; its token-order check is the original one |

Environment: 1 × H100 80GB (shared: 23.5 GB held by other processes at start), Python 3.12.3,
torch 2.15.0+cu132, transformers 5.23.0. Curated copies: [results/](results/).

## Results

From `results/summary_20261115-234047-gate.json` and the per-sample `metrics_*.jsonl`.

| | MME (20) | ScreenSpot-Pro (20) |
|---|---|---|
| Median / mean trace (new tokens) | 104 / 194 | 1,020 / 997 |
| Trace range | 57–787 | 192–2,548 |
| Truncated | 0 | 0 |
| Mean visual tokens | 473 | 5,513 (3,600–8,160; native resolution, 32 px per merged token) |
| Median visual / prompt tokens | 0.92 | 0.996 |
| Visual KV, mean | 67 MiB | 775 MiB |
| Visual share of all KV at the end of decoding (median) | 0.67 | **0.86** |
| Decode ms/token: median of samples (mean) | 33.3 (41.0) | 30.2 (42.9) |
| Max decode-phase peak | 8.58 GiB | 9.98 GiB |

- KV is 147,456 bytes per token (36 layers × 8 KV heads × 128 × 2 × bf16), as predicted. Weights: 8.27 GiB.
- Token order: row-major confirmed by the server's rewritten check (column-ramp correlation 0.74 row-major vs 0.03 column-major).
- **ScreenSpot-Pro prompt caveat.** The prompt was the bare instruction. 0/20 answers contain a click
  coordinate; the model wrote how-to explanations ("…Let me know if you need further clarification!").
  The 1,020-token median therefore measures explanations, not the grounding task. MME (yes/no
  questions) has short traces. The long-trace premise on real tasks is rechecked in the gate dump,
  which uses the benchmark's GUI-click prompt and records every trace length.
- **Timing caveat.** The GPU is shared. Per-sample decode ms/token varied erratically between 26 and
  101 (floor about 28), so means are inflated by contention spikes; medians are the better estimate.

## Gate decision

**G1 passed** (`gate/pass: true`), recorded 2026-11-15 with evidence `runs/E000-env-smoke/20261115-234047-gate`:
1. `token_order_row_major` true ✓
2. Median trace ≥ 300 on at least one dataset: ScreenSpot-Pro 1,020 ✓ (see the prompt caveat)
3. ScreenSpot-Pro median visual / prompt tokens 0.996 ≥ 0.50 ✓
4. No truncated sample ✓

## Conclusion

- Supported: on high-resolution screenshots, visual tokens are almost the whole prompt (5.5k on average), and
  they stay **86% of the KV even at the end of a ~1k-token trace**.
- Refuted, in the project's favour: the pre-registered hypothesis that visual KV would be "a minority of
  decode-time KV for long traces". At these resolutions and trace lengths it dominates.
- Open: whether traces are long on the *real* tasks (the ScreenSpot-Pro prompt caveat). The gate dump records this.
- Follow-up: the decode ms/token baseline fed to E005 (mean, 42.9) is inflated by GPU sharing; see E005.
