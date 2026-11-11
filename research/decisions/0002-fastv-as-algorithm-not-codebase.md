# 0002: Treat FastV as an algorithm to reimplement, not a codebase

- **Date:** 2026-11-11
- **Status:** accepted

## Context
FastV is the standard prune-once baseline. Its official repo patches the LLaMA forward for LLaVA. Its
KV-cache mode is a one-shot static prune, and the authors report only about 8% latency gain on images.

## Decision
Reimplement FastV and PDrop inside our cache with **physical deletion** of the KV, following
Reroute's `physical_delete`. Masking does not count as deletion for memory or latency.

## Consequences
FastV numbers are comparable to ours on memory and latency. We must check that our accuracy roughly
matches published FastV numbers at the same ratio (a sanity row in E002/E004).
