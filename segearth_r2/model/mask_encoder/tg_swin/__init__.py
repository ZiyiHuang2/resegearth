from .dynamic_relational_wti import EvidenceTokenFusion, StageDynamicRelationalWTI
from .text_condition_factory import StageWiseTextRouter, TextConditionFactory
from .window_text_interaction import (
    StageWTIHeadAware,
    StageWTIKeyBias,
    TGSwimController,
)

# backward compat alias
StageWTI = StageWTIHeadAware

__all__ = [
    "TextConditionFactory",
    "StageWiseTextRouter",
    "StageWTI",
    "StageWTIHeadAware",
    "StageWTIKeyBias",
    "TGSwimController",
    "EvidenceTokenFusion",
    "StageDynamicRelationalWTI",
]
