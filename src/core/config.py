"""Shared run config. Kept as a plain dataclass -- no YAML/pydantic needed.
Technique-specific tunables (e.g. LMCache's chunk size) belong on that
Technique's own __init__ instead of here; see techniques/lmcache_offload.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    model: str = "Qwen/Qwen2.5-1.5B"
    contexts: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192])
    max_tokens: int = 64
    gpu_utilization: float = 0.85
    output: str | None = None
