"""
sdk_utils.py — Foundry Local SDK Helpers
==========================================
Centralizes SDK initialization and model loading logic so that
ingest.py and app.py don't duplicate boilerplate code.

Usage:
    from sdk_utils import init_sdk, load_model
"""

from foundry_local_sdk import Configuration, FoundryLocalManager


def init_sdk(app_name: str) -> FoundryLocalManager:
    """
    Initializes the Foundry Local SDK singleton.

    Safe to call multiple times — if the SDK is already initialized
    (e.g., when Streamlit hot-reloads or switches models in the same
    process), the existing instance is returned without error.

    Parameters:
        app_name : Identifier registered with the Foundry Local service.

    Returns:
        The active FoundryLocalManager singleton instance.
    """
    try:
        config = Configuration(app_name=app_name)
        FoundryLocalManager.initialize(config)
    except Exception as exc:
        # Silently ignore "already initialized" — all other exceptions re-raised.
        if "already been initialized" not in str(exc):
            raise
    return FoundryLocalManager.instance


def load_model(manager: FoundryLocalManager,
               model_name: str,
               model_type: str = "model",
               progress_callback=None):
    """
    Downloads (if not cached) and loads a Foundry Local model into memory.

    Parameters:
        manager           : Active FoundryLocalManager instance.
        model_name        : Catalog alias (e.g. 'qwen3-embedding-0.6b').
        model_type        : Human-readable label used in log output.
        progress_callback : Optional callable(float) receiving download
                            progress as a percentage (0.0–100.0).
                            Pass None to suppress all progress output.

    Returns:
        The loaded Foundry Local model object.
    """
    model = manager.catalog.get_model(model_name)

    # download() is idempotent — skips network transfer when already cached.
    cb = progress_callback if progress_callback is not None else lambda p: None
    model.download(cb)

    model.load()
    return model
