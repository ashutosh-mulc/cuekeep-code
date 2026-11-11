# E000: Server environment smoke test + trace-length census

- **Created:** 2026-11-11 · **Status:** ☐ planned · **Resolves:** M1 setup
- **Config:** [config.yaml](config.yaml) · **Provenance:** [provenance.yaml](provenance.yaml)

## Pre-registration

**Question.** Does Qwen3-VL-4B-Thinking load and generate on the server with the pinned stack? How
long are its thinking traces, and how much of the KV cache do visual tokens take up?

**Hypothesis.** Traces are 1k–8k tokens. At high resolution, the visual tokens are the larger share
of the prompt KV.

**Success criterion.** 20 MME samples and 20 ScreenSpot-Pro samples generate without error. We
record per-sample visual tokens, merged grid (h, w), trace length, peak memory and ms/token. We also
confirm and pin the ScreenSpot-Pro field names.

**Setup.** 1 GPU, bf16, SDPA, greedy decoding, `max_new_tokens=8192`.

## Runs

| Run dir | Date | Tag / overrides | Notes |
|---|---|---|---|
| | | | |

## Results

## Conclusion
