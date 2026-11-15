# E002: Full-KV reference on the thinking suite

- **Created:** 2026-11-11 · **Status:** ☐ planned · **Resolves:** gate G6; Q11; the reference row of every table
- **Provenance:** [provenance.yaml](provenance.yaml)
- **Tiers:** gate G6 ([config.gate.yaml](config.gate.yaml), requires G1) → main ([config.main.yaml](config.main.yaml), requires G6).
- Pre-registration revised 2026-11-12 (decoding and seeds) and 2026-11-16 (in-repo harness on two
  benchmarks; [decisions/0006](../../research/decisions/0006-prepared-ahead-g4-g5-g6.md)), before any run.

## Pre-registration

**Question.** What are the full-KV accuracy, trace length, total response time, decode ms/token and
memory of Qwen3-VL-4B-Thinking? Every later method is compared with these.

**Hypothesis.** Our harness lands near the officially reported scores (Qwen3-VL-4B-Thinking model card:
MathVista-mini 79.5, ScreenSpot-Pro 49.2), within sampling noise and the cost of rule-based scoring.

**Setup (both tiers).** In-repo harness (`experiments/E002-full-kv-reference/run.py`):
- The model's own decoding (sampling: T 1.0, top_p 0.95, top_k 20), `max_new_tokens` 40960, per-sample seeds.
- MathVista testmini, prompted with its `query` field. Rule-based scoring of the text after `</think>`
  (`cuekeep.eval.scoring`): option letter → choice text, integer, float at the stated precision, list.
- ScreenSpot-Pro from the local release, with VLMEvalKit's GUI-click prompt. A hit is the clicked
  point inside the box, reading coordinates as Qwen3-VL's relative 0–1000. Hits under the pixel
  convention are also recorded.

### Gate tier (G6)

- **Sample.** The dev split: 150 samples per benchmark, 1 seed.
- **Pass criteria** (computed into `summary.json` as `gate/*`):
  1. Both benchmarks run end to end.
  2. Neither is more than 8 points below its official score. At n = 150 one score has a standard
     error of about 4 points, so this catches a broken harness, not small gaps.
  3. Truncation rate < 5% on both.
  4. A GPU-hour estimate for the main tier is computed (`main_tier_gpu_hours_estimate`).
- **On fail.** Look at the answers (`samples.jsonl`): fix prompting or answer extraction, and rerun.
  Do not start the main tier.
- **Unlocks.** The main tier; gate G7.

### Main tier (held by G6)

- **Requires gates.** G6, and the human's approval of the cost estimate.
- **Sample.** The full benchmarks (MathVista testmini 1,000; ScreenSpot-Pro 1,581), base seeds {0, 1, 2}.
- **Success criteria.** Mean ± std over seeds per benchmark, for the full set and for the test split
  alone, next to the official scores. Trace-length distributions are saved. The across-seed std is
  recorded under Q11. Further suite benchmarks (MMMU, V*, HRBench) need loaders and scorers first.

## Runs

| Run dir | Tier | Date | Overrides | Notes |
|---|---|---|---|---|
| | | | | |

## Results

## Gate decision

## Conclusion
