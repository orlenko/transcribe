"""Path helpers for transcribe-local."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_HOME_DIRNAME = ".transcribe_local"
TRANSCRIBE_HOME_ENV = "TRANSCRIBE_LOCAL_HOME"


def get_data_dir() -> Path:
    """Return the transcribe-local data directory."""
    custom_home = os.environ.get(TRANSCRIBE_HOME_ENV)
    if custom_home:
        return Path(custom_home).expanduser().resolve()
    return (Path.home() / DEFAULT_HOME_DIRNAME).resolve()


def ensure_data_dir() -> Path:
    """Ensure and return the data directory."""
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_env_path() -> Path:
    """Return the expected .env file path."""
    return get_data_dir() / ".env"
