"""Add a new technique by dropping one file in src/techniques/ decorated
with @register("name") -- no other file needs to change. main.py and this
registry discover it automatically via the auto-import in
techniques/__init__.py.

Convention: all Technique.__init__ parameters MUST have defaults (zero-arg
convention). This lets the registry instantiate any technique with no args,
and lets --technique-args override specific values from the CLI without
needing a custom factory per technique. If you add a required param, the
registry will raise TypeError at startup -- catch it early by running
`uv run main.py --technique <name> --dry-run`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .technique import Technique

_REGISTRY: dict[str, type["Technique"]] = {}


def register(name: str):
    def decorator(cls: type["Technique"]):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def _ensure_loaded() -> None:
    import src.techniques  # noqa: F401 -- import side effect populates _REGISTRY


def get_technique(name: str, **kwargs: Any) -> "Technique":
    """Look up and instantiate a technique by name.

    kwargs are forwarded to the technique's __init__ so that CLI
    --technique-args overrides work without changing the technique file.
    Any param not supplied falls back to the __init__ default.
    """
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown technique '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_techniques() -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY)
