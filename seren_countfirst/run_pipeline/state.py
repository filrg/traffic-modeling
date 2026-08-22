from __future__ import annotations


from ..paths import RESULTS, STATE_DIR
from ..state import load_portable_state, rebuild_model, save_portable_state


def load_state() -> dict:
    return load_portable_state(STATE_DIR)


def save_state(state: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    save_portable_state(state, STATE_DIR)


def fitted_models(state: dict) -> dict:
    """Return every fitted model, rebuilt from the portable state; never pickle."""
    models = {name: rebuild_model(entry) for name, entry in state.get("models_json", {}).items()}
    models.update(state.get("models", {}))
    return models
