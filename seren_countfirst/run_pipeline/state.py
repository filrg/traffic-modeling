from __future__ import annotations

from typing import TYPE_CHECKING

from ..state import PipelineState, rebuild_model

if TYPE_CHECKING:
    from ..models import CountModel


def fitted_models(state: PipelineState) -> dict[str, CountModel]:
    """Return every fitted model, rebuilt from the portable state; never pickle."""
    models = {name: rebuild_model(entry) for name, entry in state.get("models_json", {}).items()}
    models.update(state.get("models", {}))
    return models
