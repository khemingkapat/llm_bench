from src.core.axis import Axis
from src.core.registry import register
from src.core.technique import Technique


@register("flexgen_style")
class FlexGenStyleTechnique(Technique):
    """Approximates FlexGen's offload strategy using vLLM's built-in knobs:
    KV cache swapped to CPU, some model weights offloaded to CPU, KV cache
    quantized to fp8. Touches SPATIAL (where data lives) and STRUCTURAL
    (how it's represented) -- not TEMPORAL, since vLLM's swap space doesn't
    do priority-based prefetch scheduling the way the real FlexGen does.

    This is the whole point of the axis tags: this technique and a real
    FlexGen reproduction would report the same axes, so a later comparison
    isn't "flexgen vs baseline", it's "how much of the spatial+structural
    budget did each implementation actually capture".
    """

    axes = Axis.SPATIAL | Axis.STRUCTURAL

    def __init__(self, swap_space_gb: int = 4, cpu_offload_gb: int = 2):
        self.swap_space_gb = swap_space_gb
        self.cpu_offload_gb = cpu_offload_gb

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict:
        return {
            "swap_space": self.swap_space_gb,
            "cpu_offload_gb": self.cpu_offload_gb,
            "kv_cache_dtype": "fp8",
        }
