from src.core.axis import Axis
from src.core.registry import register
from src.core.technique import Technique
from src.core.workload import ColdWarm, Workload


@register("lmcache")
class LMCacheTechnique(Technique):
    """CPU offload of the KV cache via LMCache, with prefix reuse across
    requests. Uses ColdWarm since the whole point is a cheap warm run
    after a cold prefill. Touches SPATIAL (cache moves to CPU) and
    TEMPORAL (reuse across requests) -- not STRUCTURAL, since the cache
    representation itself is unchanged.

    LMCache needs some import-order patching to work around a C-extension
    version mismatch on newer CUDA. That patching lives in setup(), which
    only runs when this technique is actually selected -- importing this
    file (which happens for every technique on every run, via the
    auto-import in techniques/__init__.py) never has side effects on its
    own. This is what was fragile in the old repo: the patches lived at
    module level, so they only worked by accident of import order in
    main.py.
    """

    axes = Axis.SPATIAL | Axis.TEMPORAL

    def __init__(self, config_path: str = "src/config/lmcache_config.yaml"):
        self.config_path = config_path

    def setup(self) -> None:
        import os
        import sys
        import types

        os.environ["LMCACHE_CONFIG_FILE"] = self.config_path

        # Work around lmcache's C-extension symbol mismatch on CUDA 12/13.
        sys.modules.setdefault("lmcache.c_ops", types.ModuleType("lmcache.c_ops"))

        import torch.distributed  # noqa: F401

        try:
            import torch.distributed._tensor as _tensor_compat

            sys.modules.setdefault("torch.distributed.tensor", _tensor_compat)
        except ImportError:
            pass

        import vllm

        vllm.__version__ = "0.6.1.post2"  # satisfies lmcache_vllm's version check

        import lmcache  # noqa: F401 -- must import after the patches above
        from lmcache_vllm.vllm_injection import InitLMCacheEnvironment  # noqa: F401

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict:
        return {}

    def workload(self) -> Workload:
        return ColdWarm()
