"""Load settings from YAML and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_settings() -> dict[str, Any]:
    return {
        "flow_idle_timeout_seconds": 120,
        "ml": {
            "mode": "auto",
            "contamination": 0.1,
            "random_state": 42,
            "model_dir": "data/models",
            "suspicious_score_threshold": 0.55,
        },
        "labels": {"path": None},
        "threat_intelligence": {"source": None},
        "database": {"path": "data/processed/ettd.db"},
        "detection_rules_path": "config/detection_rules.yaml",
    }


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None or path == "":
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_settings(settings_path: str | Path | None = None) -> dict[str, Any]:
    """Merge defaults, YAML file, and environment overrides."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = _default_settings()

    path = settings_path or os.environ.get("ETTD_SETTINGS")
    if path is None:
        example = PROJECT_ROOT / "config" / "settings.example.yaml"
        custom = PROJECT_ROOT / "config" / "settings.yaml"
        path = custom if custom.exists() else example

    cfg_path = resolve_path(path)
    if cfg_path and cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        _deep_update(settings, loaded)

    if os.environ.get("ETTD_DB_PATH"):
        settings.setdefault("database", {})["path"] = os.environ["ETTD_DB_PATH"]
    if os.environ.get("ETTD_LABELS_PATH"):
        settings.setdefault("labels", {})["path"] = os.environ["ETTD_LABELS_PATH"]
    if os.environ.get("ETTD_ML_MODE"):
        settings.setdefault("ml", {})["mode"] = os.environ["ETTD_ML_MODE"]

    return settings


def db_path(settings: dict[str, Any] | None = None) -> Path:
    settings = settings or load_settings()
    path = resolve_path(settings.get("database", {}).get("path", "data/processed/ettd.db"))
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
