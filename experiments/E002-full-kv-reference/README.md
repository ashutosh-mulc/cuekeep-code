# E002: Full-KV reference on the thinking suite

- **Created:** 2026-11-11 · **Status:** ☐ planned · **Resolves:** M1 reference numbers
- **Config:** [config.yaml](config.yaml) · **Provenance:** [provenance.yaml](provenance.yaml)

## Pre-registration

**Question.** What are the full-KV accuracy, trace length, total response time, ms/token and peak
memory of Qwen3-VL-4B-Thinking on the thinking suite? Every later method is compared with these.

**Hypothesis.** The accuracy is within ±2 points of the officially reported numbers where they exist.
If not, our harness (prompting or answer extraction) is wrong.

**Success criterion.** Numbers on every suite benchmark, plus a sanity comparison against the
officially reported scores. Trace-length distributions are saved (they determine the paging budget).

**Setup.** VLMEvalKit with our model wrapper, greedy decoding, `max_new_tokens` from E000.
Benchmarks are listed in `configs/eval/thinking_suite.yaml`; check the names against the installed
VLMEvalKit.

## Runs

| Run dir | Date | Tag / overrides | Notes |
|---|---|---|---|
| | | | |

## Results

## Conclusion
