from src.core.axis import NONE
from src.core.registry import register
from src.core.technique import Technique


@register("baseline")
class BaselineTechnique(Technique):
    """No optimization applied -- the reference point every other
    technique is measured against. Touches none of the three axes by
    design."""

    axes = NONE

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict:
        return {}
