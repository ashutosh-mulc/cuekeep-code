"""PagedVisualCache: GPU-resident pooled stand-ins + capped slot pool + pinned host store.

Status (2026-11-12): storage, paging policy, the fetch/commit protocol and attention-input assembly
are implemented and unit-tested on CPU. The CUDA stream/event path is written but not yet run on a
GPU. The hookup to HF transformers 5.x attention is in `hf_adapter.py` (stub, roadmap M1).

How this maps onto HF attention
-------------------------------
For each paged layer, the key/value sequence handed to attention is

    [ text/generated tokens | R pooled stand-ins | C slots × max_m exact tokens ]

with an additive float mask row of

    [ 0 ...                 | log m_R, or -inf if R is committed | 0 for valid slot tokens, -inf otherwise ]

so eager and SDPA attention compute exactly `pooling.attend_mixed`, with no kernel changes.

Fetch/commit protocol (per layer, per decode step t)
----------------------------------------------------
1. `step(layer, q, t, other_lse)`: score the regions, plan fetches and evictions. An evicted region's
   stand-in is re-enabled at once (on the compute stream). Each fetch is issued as an async H2D copy
   on a side stream, which first waits for compute already queued, so a reused slot is never
   overwritten while an earlier attention still reads it. Fetches stay *pending*.
2. `commit(layer, t)`: every pending fetch with `ready_step <= t` becomes visible. The compute
   stream waits on that fetch's CUDA event (a GPU-side wait, with no host sync), then the slot length
   and the committed bit are set.
3. `attention_inputs(layer)`: K, V and bias, built only from committed regions.
With `lookahead = 1`, a fetch decided at step t is used from step t+1, so its copy overlaps step t's
compute. With `lookahead = 0`, it is used in the same step, and the GPU stalls until the copy lands.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .pooling import PooledRegions, demand_scores, pool_regions, region_log_shares


# ----------------------------------------------------------------------------- policy

@dataclass
class SlotPool:
    """Decides which regions hold GPU slots under a hard cap of `capacity` regions.

    Pure bookkeeping: no tensors are moved here. Scores must be comparable across steps (use
    per-head-normalised log shares, as `PagedVisualCache.step` does).

    Policy:
    - Candidates are the non-resident regions in descending order of current score, at most
      `fetch_budget` per decision.
    - If no slot is free, the victim is the evictable resident region with the lowest EMA score. A
      region is evictable if it was fetched in an *earlier* step, and at least `min_residency` steps ago.
    - The newcomer replaces the victim only if its *current* score beats the victim's current
      score by `admit_margin`. Both sides are on the same scale, so a region fetched this step is
      never evicted again in the same step.
    """

    capacity: int
    fetch_budget: int
    ema_decay: float = 0.9
    min_residency: int = 4
    admit_margin: float = 0.0
    region_to_slot: Dict[int, int] = field(default_factory=dict)
    fetched_at: Dict[int, int] = field(default_factory=dict)
    ema: Optional[Tensor] = None

    def free_slots(self) -> List[int]:
        used = set(self.region_to_slot.values())
        return [s for s in range(self.capacity) if s not in used]

    def plan(self, scores: Tensor, step: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """scores: [R]. Returns (fetches as (region, slot), evictions as (region, slot))."""
        scores = scores.detach().float().cpu()
        self.ema = scores.clone() if self.ema is None else self.ema_decay * self.ema + (1 - self.ema_decay) * scores

        candidates = [int(r) for r in torch.argsort(scores, descending=True) if int(r) not in self.region_to_slot]
        fetches: List[Tuple[int, int]] = []
        evictions: List[Tuple[int, int]] = []
        free = self.free_slots()
        for r in candidates[: self.fetch_budget]:
            if not free:
                victim = self._pick_victim(step)
                if victim is None or scores[r].item() <= scores[victim].item() + self.admit_margin:
                    break                       # candidates are sorted, so no later one can do better
                slot = self.region_to_slot.pop(victim)
                self.fetched_at.pop(victim, None)
                evictions.append((victim, slot))
                free.append(slot)
            slot = free.pop(0)
            self.region_to_slot[r] = slot
            self.fetched_at[r] = step
            fetches.append((r, slot))
        return fetches, evictions

    def _pick_victim(self, step: int) -> Optional[int]:
        evictable = [r for r, t in self.fetched_at.items() if t < step and step - t >= self.min_residency]
        if not evictable:
            return None
        return min(evictable, key=lambda r: self.ema[r].item())

    def resident_mask(self, num_regions: int) -> Tensor:
        """Regions holding a slot (committed or pending), as the planner sees them."""
        m = torch.zeros(num_regions, dtype=torch.bool)
        if self.region_to_slot:
            m[list(self.region_to_slot)] = True
        return m


# ----------------------------------------------------------------------------- storage

@dataclass
class PendingFetch:
    region: int
    slot: int
    length: int
    ready_step: int
    event: Any = None                     # torch.cuda.Event on CUDA, None on CPU


@dataclass
class LayerVisualStore:
    pooled: PooledRegions                 # on device
    host_k: Tensor                        # [n_vis, kvh, d] CPU, tokens grouped by region; pinned on CUDA
    host_v: Tensor
    region_start: List[int]               # offset of each region in host_k / host_v
    region_len: List[int]
    slot_k: Tensor                        # [C, max_m, kvh, d] on device
    slot_v: Tensor
    slot_len: Tensor                      # [C] valid tokens per slot; 0 while empty or pending
    committed: Tensor                     # [R] bool on device: exact KV usable by attention
    pending: List[PendingFetch] = field(default_factory=list)


class PagedVisualCache:
    """Per-layer visual KV store. Text KV stays in the HF cache; this class owns only visual KV.

    Multiple images: concatenate their visual K/V in sequence order, build ids with
    `pooling.region_ids_for_spans`, and ingest once per layer.
    """

    def __init__(self, cfg: dict, device: torch.device | str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.layers: Dict[int, LayerVisualStore] = {}
        self.pools: Dict[int, SlotPool] = {}
        self.bytes_h2d = 0                      # PCIe traffic accounting, for metrics
        self.lookahead = int(cfg.get("lookahead", 1))
        self._cuda = self.device.type == "cuda"
        self._copy_stream = torch.cuda.Stream(device=self.device) if self._cuda else None

    # -- prefill ---------------------------------------------------------------------------
    def ingest(self, layer: int, k: Tensor, v: Tensor, region_ids: Tensor) -> None:
        """k, v: [kvh, n_vis, d], all visual KV of the request at `layer` (batch size 1)."""
        if layer in self.layers:
            raise ValueError(f"layer {layer} already ingested; pass all images in one call (region_ids_for_spans)")
        R = int(region_ids.max()) + 1
        pooled = pool_regions(k, v, region_ids.to(k.device), R, cov_rank=self.cfg.get("cov_rank", 0))
        rid = region_ids.to(k.device)
        order = torch.argsort(rid, stable=True)
        counts = pooled.counts.cpu()
        starts = (torch.cumsum(counts, 0) - counts).tolist()
        lens = counts.tolist()
        host_k = k[:, order].permute(1, 0, 2).contiguous().cpu()       # [n, kvh, d], region-contiguous
        host_v = v[:, order].permute(1, 0, 2).contiguous().cpu()
        if self._cuda:
            host_k, host_v = host_k.pin_memory(), host_v.pin_memory()
        kvh, _, d = k.shape
        C, max_m = self.cfg["slot_capacity"], max(lens)
        self.layers[layer] = LayerVisualStore(
            pooled=pooled, host_k=host_k, host_v=host_v, region_start=starts, region_len=lens,
            slot_k=k.new_zeros(C, max_m, kvh, d),
            slot_v=v.new_zeros(C, max_m, kvh, v.shape[-1]),
            slot_len=torch.zeros(C, dtype=torch.long, device=k.device),
            committed=torch.zeros(R, dtype=torch.bool, device=k.device),
        )
        self.pools[layer] = SlotPool(
            capacity=C,
            fetch_budget=self.cfg.get("fetch_budget", C),
            ema_decay=self.cfg.get("ema_decay", 0.9),
            min_residency=self.cfg.get("min_residency", 4),
            admit_margin=self.cfg.get("admit_margin", 0.0),
        )

    # -- decode ----------------------------------------------------------------------------
    def region_scores(self, layer: int, q: Tensor, other_lse: Optional[Tensor] = None) -> Tensor:
        """Aggregated per-region log share [R] for queries q [heads, nq, d].

        other_lse [heads, nq]: log-sum-exp of the query's logits over all non-visual keys (text and
        generated tokens), which the attention wrapper computes from the text KV. Shares are
        normalised per head before aggregating, so heads with large logits that attend to text do
        not dominate.
        """
        st = self.layers[layer]
        heads, nq, d = q.shape
        kvh = st.pooled.keys.shape[0]
        grp = heads // kvh
        qg = q.reshape(kvh, grp * nq, d)                    # GQA: query head h uses KV head h // grp
        lse = other_lse.reshape(kvh, grp * nq) if other_lse is not None else None
        s = demand_scores(qg, st.pooled, scale=d ** -0.5, cov_mode=self.cfg.get("cov_mode", "capped"))
        shares = region_log_shares(s, lse)                  # [kvh, grp*nq, R]
        flat = shares.reshape(-1, shares.shape[-1])
        if self.cfg.get("score_agg", "max") == "max":
            return flat.amax(0)
        return torch.logsumexp(flat, 0) - math.log(flat.shape[0])     # log of the mean share

    def step(self, layer: int, q: Tensor, step: int, other_lse: Optional[Tensor] = None) -> None:
        """Score the regions, then plan and issue fetches and evictions (see the module docstring)."""
        st = self.layers[layer]
        fetches, evictions = self.pools[layer].plan(self.region_scores(layer, q, other_lse), step)
        for region, slot in evictions:
            st.pending = [p for p in st.pending if p.region != region]
            st.committed[region] = False                     # stand-in back in use immediately
            st.slot_len[slot] = 0
        if not fetches:
            return
        if self._cuda:
            # Copies into reused slots must not start before compute already queued has read them.
            self._copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        for region, slot in fetches:
            m = st.region_len[region]
            a = st.region_start[region]
            k_src, v_src = st.host_k[a:a + m], st.host_v[a:a + m]   # contiguous (pinned) views: no gather
            event = None
            if self._cuda:
                with torch.cuda.stream(self._copy_stream):
                    st.slot_k[slot, :m].copy_(k_src, non_blocking=True)
                    st.slot_v[slot, :m].copy_(v_src, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
            else:
                st.slot_k[slot, :m].copy_(k_src)
                st.slot_v[slot, :m].copy_(v_src)
            st.pending.append(PendingFetch(region, slot, m, step + self.lookahead, event))
            self.bytes_h2d += k_src.numel() * k_src.element_size() + v_src.numel() * v_src.element_size()

    def commit(self, layer: int, step: int) -> List[int]:
        """Make the fetches due by `step` visible to attention. Returns the regions committed."""
        st = self.layers[layer]
        done, keep = [], []
        for p in st.pending:
            if p.ready_step > step:
                keep.append(p)
                continue
            if p.event is not None:
                torch.cuda.current_stream(self.device).wait_event(p.event)   # GPU-side wait, no host sync
            st.slot_len[p.slot] = p.length
            st.committed[p.region] = True
            done.append(p.region)
        st.pending = keep
        return done

    # -- attention inputs ------------------------------------------------------------------
    def attention_inputs(self, layer: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Visual part of K and V, [kvh, R + C·max_m, d] each, and the additive float32 bias [R + C·max_m]."""
        st = self.layers[layer]
        C, max_m, kvh, d = st.slot_k.shape
        pooled_bias = st.pooled.log_m.to(st.committed.device).masked_fill(st.committed, float("-inf"))
        pos = torch.arange(max_m, device=st.slot_len.device)
        valid = pos[None, :] < st.slot_len[:, None]
        slot_bias = torch.zeros(valid.shape, dtype=torch.float32, device=valid.device).masked_fill(~valid, float("-inf"))
        k = torch.cat([st.pooled.keys, st.slot_k.permute(2, 0, 1, 3).reshape(kvh, C * max_m, d)], dim=1)
        v = torch.cat([st.pooled.values, st.slot_v.permute(2, 0, 1, 3).reshape(kvh, C * max_m, -1)], dim=1)
        return k, v, torch.cat([pooled_bias, slot_bias.reshape(-1)])

    # -- accounting ------------------------------------------------------------------------
    def gpu_bytes(self) -> int:
        """Device bytes held for visual KV: pooled table, covariance factors and slot buffers."""
        total = 0
        for st in self.layers.values():
            for t in (st.pooled.keys, st.pooled.values, st.pooled.cov_basis, st.slot_k, st.slot_v):
                if t is not None:
                    total += t.numel() * t.element_size()
        return total
