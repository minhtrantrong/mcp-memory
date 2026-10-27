"""Provider and model selection shared by the MCP server and the dashboard.

The dashboard writes a small JSON file so a provider or model change made in the
browser reaches the running server on its next tool call.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

PROVIDERS = ("lmstudio", "ollama", "gemini")
DEFAULT_PROVIDER = "lmstudio"
DEFAULT_MODELS = {
    "lmstudio": "liquid/lfm2.5-1.2b",
    "ollama": "qwen2.5:1.6b",
    "gemini": "gemini-3.1-flash-lite",
}
LMSTUDIO_API_BASE = "http://127.0.0.1:1234/v1"
OLLAMA_API_BASE = "http://127.0.0.1:11434"


def default_model(provider: str) -> str:
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["gemini"])


def config_path() -> Path:
    configured = os.getenv("MEMORY_CONFIG", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "memory-config.json"


def env_selection() -> tuple[str, str]:
    """Selection from environment variables, falling back to the built-in defaults."""
    provider = os.getenv("MEMORY_PROVIDER", "").strip().lower() or DEFAULT_PROVIDER
    model = os.getenv("MODEL", "").strip() or default_model(provider)
    return provider, model


def saved_selection() -> dict[str, str]:
    try:
        payload: Any = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        key: payload[key].strip()
        for key in ("provider", "model")
        if isinstance(payload.get(key), str) and payload[key].strip()
    }


def load_selection() -> tuple[str, str]:
    """Return the effective provider and model: saved choice, env vars, defaults."""
    saved = saved_selection()
    provider = saved.get("provider", "").lower()
    if provider not in PROVIDERS:
        return env_selection()
    return provider, saved.get("model", "") or default_model(provider)


def save_selection(provider: str, model: str) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provider": provider, "model": model}
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def discover_models(provider: str, timeout: float = 1.5) -> list[str]:
    """List the model ids a local provider is serving; empty when unreachable."""
    if provider == "lmstudio":
        url = f"{os.getenv('LMSTUDIO_API_BASE', LMSTUDIO_API_BASE).rstrip('/')}/models"
        entry_key = "id"
    elif provider == "ollama":
        url = f"{os.getenv('OLLAMA_API_BASE', OLLAMA_API_BASE).rstrip('/')}/api/tags"
        entry_key = "name"
    else:
        return []

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload: Any = json.loads(response.read().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return []

    entries = payload.get("data" if provider == "lmstudio" else "models")
    if not isinstance(entries, list):
        return []
    models = [
        entry[entry_key]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get(entry_key), str)
    ]
    return sorted(dict.fromkeys(models))
