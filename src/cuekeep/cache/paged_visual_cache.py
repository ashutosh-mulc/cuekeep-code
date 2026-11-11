"""PagedVisualCache: GPU-resident pooled stand-ins + capped slot pool + pinned host store.

Status (2026-11-11): the storage, the paging policy and the attention-input assembly are
implemented and unit-tested on CPU. The hookup to HF transformers 5.x (`Cache` subclass + additive
4D mask) is in `hf_adapter.py`. It is still a stub, to be written against the transformers version
installed on the GPU server (roadmap M1).

How this maps onto HF attention
-------------------------------
For each paged layer, the key/value sequence handed to attention is

    [ text/generated tokens | R pooled stand-ins | C slots × max_m exact tokens ]

with an additive mask row of

    [ 0 ...                 | log m_R, or -inf if R is resident | 0 for filled slot tokens, -inf for padding ]

so eager and SDPA attention compute exactly `pooling.attend_mixed` with no kernel changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .pooling import PooledRegions, demand_scores, pool_regions


# ----------------------------------------------------------------------------- policy

@dataclass
class SlotPool:
    """Decides which regions are resident in GPU slots under a hard cap of `capacity` regions.

    Pure bookkeeping, with no tensors moved here. The policy is: fetch the regions with the highest
    current score that are not yet resident, up to `fetch_budget` per decision. Evict the resident
    region with the lowest EMA score, unless it was fetched less than `min_residency` steps ago.
    """

    capacity: int
    fetch_budget: int
    ema_decay: float = 0.9
    min_residency: int = 4
    region_to_slot: Dict[int, int] = field(default_factory=dict)
    fetched_at: Dict[int, int] = field(default_factory=dict)
    ema: Optional[Tensor] = None

    def free_slots(self) -> List[int]:
        used = set(self.region_to_slot.values())
        return [s for s in range(self.capacity) if s not in used]

    def plan(self, scores: Tensor, step: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        """scores: [R] aggregated demand. Returns (fetches as (region, slot), evicted regions)."""
        scores = scores.detach().float().cpu()
        self.ema = scores.clone() if self.ema is None else self.ema_decay * self.ema + (1 - self.ema_decay) * scores

        candidates = [int(r) for r in torch.argsort(scores, descending=True) if int(r) not in self.region_to_slot]
        fetches: List[Tuple[int, int]] = []
        evicted: List[int] = []
        free = self.free_slots()
        for r in candidates[: self.fetch_budget]:
            if not free:
                victim = self._pick_victim(step, protect=scores[r].item())
                if victim is None:
                    break
                free.append(self.region_to_slot.pop(victim))
                self.fetched_at.pop(victim, None)
                evicted.append(victim)
            slot = free.pop(0)
            self.region_to_slot[r] = slot
            self.fetched_at[r] = step
            fetches.append((r, slot))
        return fetches, evicted

    def _pick_victim(self, step: int, protect: float) -> Optional[int]:
        evictable = [r for r, t in self.fetched_at.items() if step - t >= self.min_residency]
        if not evictable:
            return None
        victim = min(evictable, key=lambda r: self.ema[r].item())
        # Only evict if the newcomer is more in demand than the victim's smoothed score.
        return victim if self.ema[victim].item() < protect else None

    def resident_mask(self, num_regions: int) -> Tensor:
        m = torch.zeros(num_regions, dtype=torch.bool)
        if self.region_to_slot:
            m[list(self.region_to_slot)] = True
        return m


# ----------------------------------------------------------------------------- storage

@dataclass
class LayerVisualStore:
    pooled: PooledRegions                 # on device
    host_k: Tensor                        # [kvh, n_vis, d] on CPU, pinned when CUDA is available
    host_v: Tensor
    region_index: List[Tensor]            # token indices of each region (CPU)
    slot_k: Tensor                        # [C, kvh, max_m, d] on device
    slot_v: Tensor
    slot_len: Tensor                      # [C] valid tokens per slot (0 means empty)


class PagedVisualCache:
    """Per-layer visual KV store. Text KV stays in the HF cache; this class owns only visual KV."""

    def __init__(self, cfg: dict, device: torch.device | str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.layers: Dict[int, LayerVisualStore] = {}
        self.pools: Dict[int, SlotPool] = {}
        self.bytes_h2d = 0                      # PCIe traffic accounting, for metrics
        self._copy_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    # -- prefill ---------------------------------------------------------------------------
    def ingest(self, layer: int, k: Tensor, v: Tensor, region_ids: Tensor) -> None:
        """k, v: [kvh, n_vis, d] visual KV of one image at `layer` (batch size 1)."""
        R = int(region_ids.max()) + 1
        pooled = pool_regions(k, v, region_ids.to(k.device), R, cov_rank=self.cfg.get("cov_rank", 0))
        rid_cpu = region_ids.cpu()
        region_index = [(rid_cpu == r).nonzero(as_tuple=True)[0] for r in range(R)]
        max_m = max(int(ix.numel()) for ix in region_index)
        C = self.cfg["slot_capacity"]
        pin = self.device.type == "cuda"
        host_k = k.detach().to("cpu").contiguous()
        host_v = v.detach().to("cpu").contiguous()
        if pin:
            host_k, host_v = host_k.pin_memory(), host_v.pin_memory()
        kvh, _, d = k.shape
        self.layers[layer] = LayerVisualStore(
            pooled=pooled,
            host_k=host_k,
            host_v=host_v,
            region_index=region_index,
            slot_k=k.new_zeros(C, kvh, max_m, d),
            slot_v=v.new_zeros(C, kvh, max_m, v.shape[-1]),
            slot_len=torch.zeros(C, dtype=torch.long, device=k.device),
        )
        self.pools[layer] = SlotPool(
            capacity=C,
            fetch_budget=self.cfg.get("fetch_budget", C),
            ema_decay=self.cfg.get("ema_decay", 0.9),
            min_residency=self.cfg.get("min_residency", 4),
        )

    # -- decode ----------------------------------------------------------------------------
    def step(self, layer: int, q: Tensor, step: int) -> None:
        """Score regions with the current query (q: [heads, nq, d]), then fetch and evict."""
        st = self.layers[layer]
        heads, kvh = q.shape[0], st.pooled.keys.shape[0]
        qg = q.reshape(kvh, heads // kvh * q.shape[1], q.shape[-1])          # GQA: group query heads per KV head
        s = demand_scores(qg, st.pooled, scale=q.shape[-1] ** -0.5)       # [kvh, nq', R]
        agg = s.amax(dim=(0, 1)) if self.cfg.get("score_agg", "max") == "max" else s.mean(dim=(0, 1))
        fetches, _ = self.pools[layer].plan(agg, step)
        for region, slot in fetches:
            self._fetch(st, region, slot)

    def _fetch(self, st: LayerVisualStore, region: int, slot: int) -> None:
        ix = st.region_index[region]
        m = ix.numel()
        k_src = st.host_k[:, ix, :]
        v_src = st.host_v[:, ix, :]
        if self._copy_stream is not None:
            k_src, v_src = k_src.pin_memory(), v_src.pin_memory()
            with torch.cuda.stream(self._copy_stream):
                st.slot_k[slot, :, :m].copy_(k_src, non_blocking=True)
                st.slot_v[slot, :, :m].copy_(v_src, non_blocking=True)
        else:
            st.slot_k[slot, :, :m].copy_(k_src)
            st.slot_v[slot, :, :m].copy_(v_src)
        st.slot_len[slot] = m
        self.bytes_h2d += k_src.numel() * k_src.element_size() + v_src.numel() * v_src.element_size()

    def sync(self) -> None:
        """Make pending fetches visible to the compute stream (call before attention)."""
        if self._copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self._copy_stream)

    # -- attention inputs ------------------------------------------------------------------
    def attention_inputs(self, layer: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Visual part of K, V and the additive bias: [kvh, R + C·max_m, d] ×2 and [R + C·max_m]."""
        st = self.layers[layer]
        R = st.pooled.num_regions
        C, kvh, max_m, d = st.slot_k.shape
        resident = self.pools[layer].resident_mask(R).to(st.pooled.log_m.device)
        pooled_bias = st.pooled.log_m.masked_fill(resident, float("-inf"))
        pos = torch.arange(max_m, device=st.slot_len.device)
        slot_bias = torch.where(pos[None, :] < st.slot_len[:, None], 0.0, float("-inf")).reshape(-1)
        k = torch.cat([st.pooled.keys, st.slot_k.permute(1, 0, 2, 3).reshape(kvh, C * max_m, d)], dim=1)
        v = torch.cat([st.pooled.values, st.slot_v.permute(1, 0, 2, 3).reshape(kvh, C * max_m, -1)], dim=1)
        bias = torch.cat([pooled_bias.to(slot_bias.dtype), slot_bias])
        return k, v, bias

    # -- accounting ------------------------------------------------------------------------
    def gpu_bytes(self) -> int:
        total = 0
        for st in self.layers.values():
            for t in (st.pooled.keys, st.pooled.values, st.slot_k, st.slot_v, st.pooled.cov_basis):
                if t is not None:
                    total += t.numel() * t.element_size()
        return total
