from __future__ import annotations

from traffic_modeling.core import StageInvocation

from .planning import InvocationRecipe
from .registry import REPRODUCTION_STEPS

DEFAULT_INVOCATION_RECIPE = InvocationRecipe(
    name="seren-countfirst-full",
    invocations=tuple(
        StageInvocation(step.argv[0], step.argv[1:]) for step in REPRODUCTION_STEPS
    ),
)
