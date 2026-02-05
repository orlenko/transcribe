"""Configuration management for transcribe-local."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..paths import get_data_dir


@dataclass
class Settings:
    """Application settings."""

    # Paths
    data_dir: Path = field(default_factory=get_data_dir)
    db_path: Path = field(default_factory=lambda: get_data_dir() / "speakers.db")
    upload_dir: Path = field(default_factory=lambda: get_data_dir() / "uploads")
    config_path: Path = field(default_factory=lambda: get_data_dir() / "config.json")

    # Transcription settings
    transcription_mode: str = "hybrid"  # "hybrid", "openai", "local"
    openai_model: str = "whisper-1"
    local_model: str = "large-v3"
    default_language: Optional[str] = None

    # Diarization settings (always local)
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    speaker_match_threshold: float = 0.5

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    server_url: str = "http://localhost:8000"

    def _json_dict(self) -> dict[str, Any]:
        return {
            "transcription_mode": self.transcription_mode,
            "openai_model": self.openai_model,
            "local_model": self.local_model,
            "default_language": self.default_language,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "speaker_match_threshold": self.speaker_match_threshold,
            "host": self.host,
            "port": self.port,
            "server_url": self.server_url,
        }

    def save(self) -> None:
        """Save settings to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._json_dict(), f, indent=2)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from config file."""
        settings = cls()
        if config_path:
            settings.config_path = config_path

        if settings.config_path.exists():
            try:
                with open(settings.config_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}

            settings.transcription_mode = data.get("transcription_mode", settings.transcription_mode)
            settings.openai_model = data.get("openai_model", settings.openai_model)
            settings.local_model = data.get("local_model", settings.local_model)
            settings.default_language = data.get("default_language", settings.default_language)
            settings.min_speakers = data.get("min_speakers", settings.min_speakers)
            settings.max_speakers = data.get("max_speakers", settings.max_speakers)
            settings.speaker_match_threshold = data.get("speaker_match_threshold", settings.speaker_match_threshold)
            settings.host = data.get("host", settings.host)
            settings.port = data.get("port", settings.port)
            settings.server_url = data.get("server_url", settings.server_url)

        return settings


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings
