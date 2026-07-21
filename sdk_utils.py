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
    
    # Check if the model is already on disk
    was_cached = model.is_cached
    
    if not was_cached:
        cb = progress_callback if progress_callback is not None else lambda p: None
        model.download(cb)

    model.load()
    return model


# ------------------------------------------------------------------
# Background Memory Monitor
# ------------------------------------------------------------------
import threading
import time

class MemoryMonitor:
    last_query_time = time.time()
    timeout_seconds = 30  # Default to 30 seconds
    _thread = None
    is_busy = False

    @classmethod
    def start(cls):
        """Starts the background thread if it isn't running already."""
        if cls._thread is None:
            cls._thread = threading.Thread(target=cls._watchdog, daemon=True)
            cls._thread.start()

    @classmethod
    def ping(cls):
        """Resets the idle timer (call this whenever the user interacts)."""
        cls.last_query_time = time.time()

    @classmethod
    def set_busy(cls, busy: bool):
        """Sets the busy flag to prevent unloading while actively working."""
        cls.is_busy = busy
        if not busy:
            # When finishing work, immediately ping so we get a full idle window
            cls.ping()

    @classmethod
    def force_unload_after(cls, delay_seconds: int):
        """Forcefully unloads all models after a specific delay, ignoring standard timeout locks."""
        def _unload():
            time.sleep(delay_seconds)
            from sdk_utils import FoundryLocalManager
            if FoundryLocalManager.instance and hasattr(FoundryLocalManager.instance, "catalog"):
                try:
                    for m in FoundryLocalManager.instance.catalog.get_loaded_models():
                        if hasattr(m, 'unload'):
                            m.unload()
                except Exception:
                    pass
        threading.Thread(target=_unload, daemon=True).start()

    @classmethod
    def set_timeout(cls, seconds: int):
        """Updates the auto-free timeout."""
        cls.timeout_seconds = seconds
        # Also ping immediately so changing the dropdown doesn't instantly unload if idle
        cls.ping()

    @classmethod
    def _watchdog(cls):
        """Background loop that checks for idle expiration."""
        while True:
            time.sleep(5)
            # Skip checking if the app is actively processing something
            if cls.is_busy:
                continue

            if cls.timeout_seconds > 0:
                elapsed = time.time() - cls.last_query_time
                if elapsed >= cls.timeout_seconds:
                    # Time has expired, unload models
                    if FoundryLocalManager.instance and hasattr(FoundryLocalManager.instance, "catalog"):
                        try:
                            models = FoundryLocalManager.instance.catalog.get_loaded_models()
                            for m in models:
                                if hasattr(m, 'unload'):
                                    m.unload()
                        except Exception:
                            pass
                    
                    # Clear Streamlit cache safely from background thread
                    try:
                        import streamlit as st
                        st.cache_resource.clear()
                    except Exception:
                        pass
                    
                    # Reset timer so we don't spam unload in a loop
                    cls.last_query_time = time.time()
