# Open questions and risks

Each item names the experiment or action that resolves it. Close an item by striking it through and
linking the evidence.

| # | Question / risk | Severity | Resolved by | Status |
|---|---|---|---|---|
| Q1 | Does the log m pooled stand-in approximate attention well on *real* Qwen3-VL keys, given that RoPE spreads keys within a region? Pool pre- or post-RoPE? | High | E001 | open |
| Q2 | Does the covariance correction track true region mass better than the plain pooled logit? What rank r is enough? | High | E001 | open |
| Q3 | Do the regions a thinking trace needs actually shift over decoding (H2)? | High | E003 (attention-shift analysis) | open |
| Q4 | Can PCIe fetch latency be hidden with a 1-step lookahead at batch size 1? | Medium | M2 profiling | open |
| Q5 | Does degraded perception lengthen traces enough to cancel the per-token savings (the ViMoD effect)? | Medium | Report total response time in every table | open |
| Q6 | Reroute license: can we copy its code, or only take inspiration? | Medium | Check the repo | open |
| Q7 | Does Vortex/SGLang support a Qwen3-VL decode path? | Low (optional) | 1-day smoke test | open |
| Q8 | Will ViMoD, VisionPulse or SparseVILA release code before reviews? | Medium | Re-check Jan 2027 | open |
| Q9 | The exact demand-score definition in the original design notes | Medium | Compare with the design session | open |
