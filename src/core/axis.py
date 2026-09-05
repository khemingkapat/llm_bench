"""The three co-optimization dimensions from the project idea, made explicit
in code so every technique has to declare which of them it touches.

This is the whole point of the "co-optimization gap" framing: existing
methods (H2O, FlexGen, ...) each pick one axis and pay for it on the others.
Tagging each Technique with the axes it touches means later analysis can
group/plot results by axis instead of by filename, and makes it obvious at
a glance which combinations from the literature review haven't been tried
together yet.
"""

from __future__ import annotations

from enum import Flag, auto


class Axis(Flag):
    """Use bitwise OR to combine, e.g. Axis.SPATIAL | Axis.TEMPORAL."""

    STRUCTURAL = auto()  # how the cache is represented/compressed (quantization, eviction, sink+window)
    SPATIAL = auto()      # where the cache lives (GPU / CPU RAM / disk, offload policy)
    TEMPORAL = auto()     # when data moves or is (re)computed (prefetch, reuse, scheduling)


NONE = Axis(0)  # a technique that (by design) touches none of the three, e.g. the baseline


def axis_names(flags: Axis) -> list[str]:
    """Decompose a combined Axis flag into its member names, e.g.
    Axis.SPATIAL | Axis.TEMPORAL -> ["SPATIAL", "TEMPORAL"]."""
    return [a.name for a in Axis if a in flags]
