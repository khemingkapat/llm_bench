"""GPU/CPU memory helpers plus the one result schema every technique writes
into. Core fields are always present so techniques stay comparable across
runs; `extra` is a free-form bucket for technique-specific numbers (PCIe
bytes moved, a quality score, ...) so adding those doesn't require changing
the schema for every other technique.

RSS (Resident Set Size) is the total RAM actually held in physical memory by
the process, as reported by the OS. It includes Python runtime, model weights
loaded to CPU, and the KV cache if it has been offloaded. We measure the
delta (before engine init → after workload) so that the baseline Python
overhead and shared libraries are subtracted out, isolating the technique's
own memory footprint.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def reset_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
    return round(alloc, 2), round(reserved, 2)


# ---------------------------------------------------------------------------
# Host RAM helpers
# ---------------------------------------------------------------------------

def host_rss_mb() -> float:
    """Current RSS of this process in MB. Call once before and once after
    the workload; subtract to get the delta attributable to the technique."""
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

def throughput(gen_tokens: int, decode_time_sec: float, total_time_sec: float) -> float:
    decode_tokens = gen_tokens - 1 if gen_tokens > 1 else gen_tokens
    if decode_time_sec > 0:
        return round(decode_tokens / decode_time_sec, 2)
    if total_time_sec > 0:
        return round(gen_tokens / total_time_sec, 2)
    return 0.0


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class ResultEntry:
    requested_ctx: int
    status: str           # "SUCCESS" | "FAILED"
    axes: list[str] = field(default_factory=list)   # copied from the technique for easy CSV filtering

    # Token counts
    prompt_tokens: int = 0
    generated_tokens: int = 0

    # Timing -- warm values (or the only values for SyntheticFiller)
    ttft_sec: float = 0.0
    decode_time_sec: float = 0.0
    total_time_sec: float = 0.0
    throughput_tok_per_sec: float = 0.0

    # Cold-run timing -- non-zero only for ColdWarm workloads.
    # cold_ttft_delta_sec = ttft_sec - cold_ttft_sec (negative = warm is faster)
    cold_ttft_sec: float = 0.0
    cold_total_sec: float = 0.0

    # GPU memory
    peak_alloc_mb: float = 0.0
    peak_reserved_mb: float = 0.0

    # Host RAM delta (RSS after workload − RSS before engine init).
    # Isolates the technique's CPU-side memory cost from Python/library baseline.
    host_rss_delta_mb: float = 0.0

    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cold_ttft_delta_sec(self) -> float:
        """How much faster the warm run was vs. cold (positive = warm is faster)."""
        if self.cold_ttft_sec == 0.0:
            return 0.0
        return round(self.cold_ttft_sec - self.ttft_sec, 4)
