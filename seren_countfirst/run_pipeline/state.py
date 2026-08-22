from __future__ import annotations


from ..state import PipelineState, rebuild_model
from .context import default_context


def load_state() -> PipelineState:
    return default_context().load_state()


def save_state(state: PipelineState) -> None:
    default_context().save_state(state)


def fitted_models(state: PipelineState) -> dict:
    """Return every fitted model, rebuilt from the portable state; never pickle."""
    models = {name: rebuild_model(entry) for name, entry in state.get("models_json", {}).items()}
    models.update(state.get("models", {}))
    return models
