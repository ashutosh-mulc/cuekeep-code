"""PagedVisualCache: GPU-resident pooled stand-ins + capped slot pool + pinned host store.

Status (2026-11-16): storage, paging policy, the fetch/commit protocol and attention-input assembly
are implemented and tested on CPU, MPS and (on the server) CUDA. G5 round 1 changes: persistent
device buffers (attention inputs are views; no per-step concatenation) and batched scoring across
layers (`queue_query` + `end_step`: one host sync per refresh instead of one per layer). The hookup
to HF transformers attention is in `hf_adapter.py` (stub).

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
    ema: Optional[List[float]] = None

    def free_slots(self) -> List[int]:
        used = set(self.region_to_slot.values())
        return [s for s in range(self.capacity) if s not in used]

    def plan(self, scores: Tensor, step: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """scores: [R]. Returns (fetches as (region, slot), evictions as (region, slot)).

        Runs on plain Python lists: it is called for every paged layer at every refresh, so
        element-wise tensor access (one `.item()` per region) would dominate its cost.
        """
        s = scores.detach().float().cpu()
        order = torch.argsort(s, descending=True).tolist()
        s = s.tolist()
        if self.ema is None:
            self.ema = list(s)
        else:
            a, b = self.ema_decay, 1 - self.ema_decay
            self.ema = [a * e + b * x for e, x in zip(self.ema, s)]

        fetches: List[Tuple[int, int]] = []
        evictions: List[Tuple[int, int]] = []
        free = self.free_slots()
        n = 0
        for r in order:
            if n >= self.fetch_budget:
                break
            if r in self.region_to_slot:
                continue
            n += 1
            if not free:
                victim = self._pick_victim(step)
                if victim is None or s[r] <= s[victim] + self.admit_margin:
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
        return min(evictable, key=self.ema.__getitem__)

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
    event: Any = None                     # torch.cuda.Event on CUDA, None otherwise


@dataclass
class LayerVisualStore:
    """Visual KV of one layer.

    Device memory is two persistent buffers, laid out sequence-major: rows [0, R) hold the pooled
    stand-ins and rows R + j·max_m … hold slot j. `bias` [R + C·max_m] is the matching additive
    attention bias. Fetches, commits and evictions update these in place, so building the attention
    inputs costs nothing per step (views only).
    """
    pooled: PooledRegions                 # on device (keys/values also live in buf rows [0, R))
    host_k: Tensor                        # [n_vis, kvh, d] CPU, tokens grouped by region; pinned on CUDA
    host_v: Tensor
    region_start: List[int]               # offset of each region in host_k / host_v
    region_len: List[int]
    buf_k: Tensor                         # [R + C·max_m, kvh, d] on device
    buf_v: Tensor
    bias: Tensor                          # [R + C·max_m] float32 on device
    num_regions: int
    capacity: int
    max_m: int
    committed: Tensor                     # [R] bool, CPU: exact KV usable by attention
    slot_len: Tensor                      # [C] long, CPU: valid tokens per slot; 0 while empty or pending
    pending: List[PendingFetch] = field(default_factory=list)

    @property
    def slot_k(self) -> Tensor:           # [C, max_m, kvh, d] view
        return self.buf_k[self.num_regions:].view(self.capacity, self.max_m, *self.buf_k.shape[1:])

    @property
    def slot_v(self) -> Tensor:
        return self.buf_v[self.num_regions:].view(self.capacity, self.max_m, *self.buf_v.shape[1:])

    def slot_rows(self, slot: int) -> slice:
        a = self.num_regions + slot * self.max_m
        return slice(a, a + self.max_m)


class PagedVisualCache:
    """Per-layer visual KV store. Text KV stays in the HF cache; this class owns only visual KV.

    Two ways to drive it per decode step t:
    - per layer: `step(layer, q, t, other_lse)` scores and plans that layer immediately;
    - batched (G5 round 1): `queue_query(layer, q, other_lse)` for every layer during the forward,
      then `end_step(t)` once: one batched scoring call and one host sync for all layers, every
      `refresh_every` steps. With `lookahead >= 1` this is the same protocol, since fetches decided
      at step t are only used from step t + lookahead.
    In both cases call `commit(layer, t)` before reading `attention_inputs(layer)`.

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
        self.refresh_every = int(cfg.get("refresh_every", 1))
        self._cuda = self.device.type == "cuda"
        self._copy_stream = torch.cuda.Stream(device=self.device) if self._cuda else None
        self._queued: Dict[int, tuple] = {}
        self._stacked: Optional[tuple] = None   # (layers, PooledRegions stacked over layers)

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
        C, max_m = int(self.cfg["slot_capacity"]), max(lens)
        rows = R + C * max_m
        buf_k = k.new_zeros(rows, kvh, d)
        buf_v = v.new_zeros(rows, kvh, v.shape[-1])
        buf_k[:R] = pooled.keys.permute(1, 0, 2)
        buf_v[:R] = pooled.values.permute(1, 0, 2)
        bias = torch.full((rows,), float("-inf"), dtype=torch.float32, device=k.device)
        bias[:R] = pooled.log_m.to(k.device, torch.float32)
        # The pooled table lives in the buffer rows; keep PooledRegions as views (no duplicate memory).
        pooled = PooledRegions(buf_k[:R].permute(1, 0, 2), buf_v[:R].permute(1, 0, 2), pooled.counts,
                               pooled.log_m, pooled.cov_basis)
        self.layers[layer] = LayerVisualStore(
            pooled=pooled, host_k=host_k, host_v=host_v, region_start=starts, region_len=lens,
            buf_k=buf_k, buf_v=buf_v, bias=bias, num_regions=R, capacity=C, max_m=max_m,
            committed=torch.zeros(R, dtype=torch.bool), slot_len=torch.zeros(C, dtype=torch.long),
        )
        self.pools[layer] = SlotPool(
            capacity=C,
            fetch_budget=self.cfg.get("fetch_budget", C),
            ema_decay=self.cfg.get("ema_decay", 0.9),
            min_residency=self.cfg.get("min_residency", 4),
            admit_margin=self.cfg.get("admit_margin", 0.0),
        )
        self._stacked = None

    # -- scoring ---------------------------------------------------------------------------
    def _aggregate(self, shares: Tensor) -> Tensor:
        """[..., Q, R] per-(head, query) log shares → [..., R] (max, or log of the mean share)."""
        flat = shares.reshape(*shares.shape[:-3], -1, shares.shape[-1])
        if self.cfg.get("score_agg", "max") == "max":
            return flat.amax(-2)
        return torch.logsumexp(flat, -2) - math.log(flat.shape[-2])

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
        return self._aggregate(region_log_shares(s, lse))

    def _stack(self, layers: List[int]) -> Optional[PooledRegions]:
        """Pooled tables of `layers` stacked on a leading dim, or None if their region sets differ."""
        if self._stacked is not None and self._stacked[0] == layers:
            return self._stacked[1]
        stores = [self.layers[l].pooled for l in layers]
        if any(not torch.equal(s.counts, stores[0].counts) for s in stores[1:]):
            return None
        cov = None
        if all(s.cov_basis is not None for s in stores):
            cov = torch.stack([s.cov_basis for s in stores])
        stacked = PooledRegions(torch.stack([s.keys for s in stores]), torch.stack([s.values for s in stores]),
                                stores[0].counts, stores[0].log_m, cov)
        self._stacked = (layers, stacked)
        return stacked

    # -- decode ----------------------------------------------------------------------------
    def step(self, layer: int, q: Tensor, step: int, other_lse: Optional[Tensor] = None) -> None:
        """Score this layer's regions now, then plan and issue its fetches and evictions."""
        self._plan_and_fetch(layer, self.region_scores(layer, q, other_lse).detach().float().cpu(), step)

    def queue_query(self, layer: int, q: Tensor, other_lse: Optional[Tensor] = None) -> None:
        """Hold this layer's decode query for the batched decision in `end_step` (no copy, no sync)."""
        self._queued[layer] = (q, other_lse)

    def end_step(self, step: int) -> None:
        """Every `refresh_every` steps: score all queued layers in one call, sync once, plan and fetch."""
        queued, self._queued = self._queued, {}
        if not queued or step % self.refresh_every != 0:
            return
        layers = sorted(queued)
        stacked = self._stack(layers)
        if stacked is None:                                   # region sets differ: per-layer fallback
            for l in layers:
                self.step(l, queued[l][0], step, queued[l][1])
            return
        q = torch.stack([queued[l][0] for l in layers])       # [Lq, heads, nq, d]
        Lq, heads, nq, d = q.shape
        kvh = stacked.keys.shape[1]
        qg = q.reshape(Lq, kvh, heads // kvh * nq, d)
        lse = None
        if all(queued[l][1] is not None for l in layers):
            lse = torch.stack([queued[l][1] for l in layers]).reshape(Lq, kvh, -1)
        s = demand_scores(qg, stacked, scale=d ** -0.5, cov_mode=self.cfg.get("cov_mode", "capped"))
        agg = self._aggregate(region_log_shares(s, lse)).detach().float().cpu()     # [Lq, R]: the one sync
        for i, l in enumerate(layers):
            self._plan_and_fetch(l, agg[i], step)

    def _plan_and_fetch(self, layer: int, scores_cpu: Tensor, step: int) -> None:
        st = self.layers[layer]
        fetches, evictions = self.pools[layer].plan(scores_cpu, step)
        for region, slot in evictions:
            st.pending = [p for p in st.pending if p.region != region]
            if bool(st.committed[region]):
                st.committed[region] = False
                st.bias[region] = st.pooled.log_m[region].to(torch.float32)   # stand-in back in use now
            st.slot_len[slot] = 0
            st.bias[st.slot_rows(slot)] = float("-inf")
        if not fetches:
            return
        if self._cuda:
            # Copies into reused slots must not start before compute already queued has read them.
            self._copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        for region, slot in fetches:
            m = st.region_len[region]
            a = st.region_start[region]
            k_src, v_src = st.host_k[a:a + m], st.host_v[a:a + m]   # contiguous (pinned) [m, kvh, d]: no gather
            rows = st.slot_rows(slot)
            dst = slice(rows.start, rows.start + m)
            event = None
            if self._cuda:
                with torch.cuda.stream(self._copy_stream):
                    st.buf_k[dst].copy_(k_src, non_blocking=True)
                    st.buf_v[dst].copy_(v_src, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
            else:
                st.buf_k[dst].copy_(k_src)
                st.buf_v[dst].copy_(v_src)
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
            rows = st.slot_rows(p.slot)
            st.bias[slice(rows.start, rows.start + p.length)] = 0.0
            st.bias[p.region] = float("-inf")                               # exact KV replaces the stand-in
            st.slot_len[p.slot] = p.length
            st.committed[p.region] = True
            done.append(p.region)
        st.pending = keep
        return done

    # -- attention inputs ------------------------------------------------------------------
    def attention_inputs(self, layer: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Visual K and V as [kvh, R + C·max_m, d] views, and the additive float32 bias [R + C·max_m]."""
        st = self.layers[layer]
        return st.buf_k.permute(1, 0, 2), st.buf_v.permute(1, 0, 2), st.bias

    # -- accounting ------------------------------------------------------------------------
    def gpu_bytes(self) -> int:
        """Device bytes held for visual KV: the persistent buffers (pooled rows + slots), bias, covariance."""
        total = 0
        for st in self.layers.values():
            for t in (st.buf_k, st.buf_v, st.bias, st.pooled.cov_basis):
                if t is not None:
                    total += t.numel() * t.element_size()
        return total
