# E000: Environment smoke test + premise census (gate G1)

- **Created:** 2026-11-11 · **Status:** ☐ planned · **Resolves:** gate G1, M1 setup
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
| | | | | |

## Results

## Gate decision

## Conclusion
