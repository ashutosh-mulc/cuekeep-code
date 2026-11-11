# E001: Pooled stand-in fidelity (log m bias, covariance correction)

- **Created:** 2026-11-11 · **Status:** ☐ planned · **Resolves:** Q1, Q2 ([open_questions](../../research/open_questions.md))
- **Config:** [config.yaml](config.yaml) · **Provenance:** [provenance.yaml](provenance.yaml)

## Pre-registration

**Question.** How well do the pooled stand-ins (mean K/V plus a log m bias) reproduce the attention
output over a region? Does the low-rank covariance term improve the estimate of each region's
attention mass enough to rank regions for fetching?

**Hypotheses.**
- H1a: the log m bias reduces attention-output error relative to pooling without it, at every g.
- H1b: error grows with g. At g=2 the stand-ins alone keep the relative output error below 10% in
  most layers.
- H1c: the covariance-corrected demand score ranks regions by true mass better than the plain pooled
  logit (higher Spearman ρ; higher top-B recall).

**Success criteria.**
1. With all regions pooled, at g=2, the median relative L2 error of the attention output is < 0.10.
   Report per layer, flagging layers that fail.
2. The log m bias lowers the median error vs no bias at every g, in ≥ 80% of layers.
3. Top-B recall of the true highest-mass regions (B = 8) improves by ≥ 5 points with r=4 vs r=0.

**Setup.**
- Phase A, CPU, synthetic (`source: synthetic`): Gaussian region keys with controlled within-region
  spread. This only sanity-checks the math and the metrics.
- Phase B, GPU (`source: dump`): K/V and decode queries dumped from Qwen3-VL-4B-Thinking on 50
  ScreenSpot-Pro and 50 MathVista images (`scripts/dump_visual_kv.py`), at a sample of decode steps
  across the trace. Sweep g ∈ {2, 4, 8}, r ∈ {0, 2, 4, 8}, and pre- vs post-RoPE pooling.

**Planned measurements.** These are mirrored in `provenance.yaml`:
- relative output error vs g, by layer
- the effect of the log m bias
- Spearman ρ and top-B recall of the demand score vs r
- pre- vs post-RoPE pooling

## Runs

| Run dir | Date | Tag / overrides | Notes |
|---|---|---|---|
| | | | |

## Results

## Conclusion
