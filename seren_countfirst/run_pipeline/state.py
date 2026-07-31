"""Load and save the resumable pipeline state."""
from __future__ import annotations


from ..paths import RESULTS, STATE_DIR
from ..tools.migrate_state import load_portable_state, rebuild_model, save_portable_state

def _load() -> dict:
    return load_portable_state(STATE_DIR)


def _save(state: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    save_portable_state(state, STATE_DIR)


def _models(state: dict) -> dict:
    """Every fitted model, rebuilt from the portable state (no pickle)."""
    models = {name: rebuild_model(entry) for name, entry in state.get("models_json", {}).items()}
    models.update(state.get("models", {}))
    return models
