"""Environment loading helpers for transcribe-local."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .paths import get_env_path


TRANSCRIBE_ENV_FILE_ENV = "TRANSCRIBE_LOCAL_ENV_FILE"


def load_transcribe_env(override: bool = False) -> bool:
    """Load environment variables for transcribe-local.

    Priority:
    1) TRANSCRIBE_LOCAL_ENV_FILE (explicit path)
    2) <TRANSCRIBE_LOCAL_HOME>/.env or ~/.transcribe_local/.env
    3) Default dotenv discovery fallback
    """
    explicit = os.environ.get(TRANSCRIBE_ENV_FILE_ENV)
    if explicit:
        env_path = Path(explicit).expanduser()
        if env_path.exists():
            return load_dotenv(dotenv_path=env_path, override=override)

    default_env_path = get_env_path()
    if default_env_path.exists():
        return load_dotenv(dotenv_path=default_env_path, override=override)

    return load_dotenv(override=override)
