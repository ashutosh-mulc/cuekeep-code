# E002: Full-KV reference on the thinking suite

- **Created:** 2026-11-11 · **Status:** ☐ planned · **Resolves:** gate G6; Q11; the reference row of every table
- **Provenance:** [provenance.yaml](provenance.yaml)
- **Tiers:** gate G6 ([config.gate.yaml](config.gate.yaml), requires G1) → main ([config.main.yaml](config.main.yaml), requires G6).
- Pre-registration revised 2026-11-12, before any run (decoding and seeds; staged-gate design).
- `run.py` is a stub until the VLMEvalKit wrapper is written (see [SERVER_CHECKLIST](../../SERVER_CHECKLIST.md)).

## Pre-registration

**Question.** What are the full-KV accuracy, trace length, total response time, decode ms/token and
memory of Qwen3-VL-4B-Thinking on the thinking suite? Every later method is compared with these.

**Hypothesis.** Our harness matches the officially reported scores, within sampling noise.

**Setup (both tiers).** VLMEvalKit with our model wrapper. The model's own decoding (sampling, T 1.0,
top_p 0.95, top_k 20), `max_new_tokens` 40960, per-sample seeds. Benchmarks are listed in
`configs/eval/thinking_suite.yaml`; check the names against the installed VLMEvalKit first.

### Gate tier (G6)

- **Sample.** The dev split: 150 samples per benchmark, 1 seed.
- **Pass criteria.**
  1. Every benchmark runs end to end.
  2. No benchmark is more than 8 points below its official score (where one exists). At n = 150 one
     score has a standard error of about 4 points, so this catches a broken harness, not small gaps.
  3. Truncation rate < 5% on every benchmark.
  4. A GPU-hour estimate for the main tier is written in the notebook, from this run's measured
     tokens/s and trace lengths.
- **On fail.** Fix prompting or answer extraction and rerun. Do not start the main tier.
- **Unlocks.** The main tier; gate G7.

### Main tier (held by G6)

- **Requires gates.** G6, and the human's approval of the cost estimate.
- **Sample.** The full benchmarks, base seeds {0, 1, 2}.
- **Success criteria.** Mean ± std over seeds on every benchmark, for the full set and for the test
  split alone, next to the official scores. Trace-length distributions are saved (they size the
  paging budget). The across-seed std is recorded under Q11.

## Runs

| Run dir | Tier | Date | Overrides | Notes |
|---|---|---|---|---|
| | | | | |

## Results

## Gate decision

## Conclusion
