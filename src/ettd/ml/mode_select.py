"""Select Isolation Forest vs Random Forest based on labels availability."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_LABEL_COLUMNS = {"flow_id", "label"}


def validate_labels_csv(path: str | Path | None) -> tuple[bool, str, pd.DataFrame | None]:
    """Return (ok, reason, dataframe). Labels must be 0/1 or benign/suspicious."""
    if path is None or str(path).strip() == "" or str(path).lower() == "null":
        return False, "no labels path configured", None
    p = Path(path)
    if not p.is_absolute():
        from ettd.config import PROJECT_ROOT

        p = PROJECT_ROOT / p
    if not p.exists():
        return False, f"labels file not found: {p}", None
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        return False, f"failed to read labels CSV: {exc}", None
    missing = REQUIRED_LABEL_COLUMNS - set(df.columns)
    if missing:
        return False, f"labels CSV missing columns: {sorted(missing)}", None
    if df.empty:
        return False, "labels CSV is empty", None

    normalized = df.copy()
    normalized["label"] = normalized["label"].map(_normalize_label)
    if normalized["label"].isna().any():
        return False, "labels must be 0/1 or benign/suspicious", None
    return True, "ok", normalized


def _normalize_label(value) -> int | None:
    if value in (0, 1):
        return int(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"benign", "normal", "0"}:
            return 0
        if v in {"suspicious", "malicious", "anomaly", "1"}:
            return 1
    return None


def select_model_mode(settings: dict) -> tuple[str, pd.DataFrame | None, str]:
    """
    Return (mode, labels_df_or_None, reason).

    mode is 'isolation_forest' or 'random_forest'.
    """
    configured = str(settings.get("ml", {}).get("mode", "auto")).lower()
    labels_path = settings.get("labels", {}).get("path")
    ok, reason, labels_df = validate_labels_csv(labels_path)

    if configured == "isolation_forest":
        logger.info("ML mode forced: isolation_forest (%s)", reason)
        return "isolation_forest", None, f"forced isolation_forest; labels: {reason}"
    if configured == "random_forest":
        if not ok:
            raise ValueError(f"random_forest requested but labels invalid: {reason}")
        logger.info("ML mode forced: random_forest")
        return "random_forest", labels_df, "forced random_forest with valid labels"
    # auto
    if ok:
        logger.info("ML mode auto-selected: random_forest (%s)", reason)
        return "random_forest", labels_df, reason
    logger.info("ML mode auto-selected: isolation_forest (%s)", reason)
    return "isolation_forest", None, reason
