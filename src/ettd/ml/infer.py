"""Run detection inference on flow features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ettd.ml.features_matrix import NUMERIC_FEATURE_COLUMNS, build_matrix
from ettd.ml.train import train_model

logger = logging.getLogger(__name__)


def _model_path(settings: dict[str, Any]) -> Path:
    from ettd.config import PROJECT_ROOT

    model_dir = Path(settings.get("ml", {}).get("model_dir", "data/models"))
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    return model_dir / "model.joblib"


def detect_suspicious(
    features_df: pd.DataFrame,
    settings: dict[str, Any],
    *,
    retrain: bool = True,
) -> list[dict[str, Any]]:
    """
    Return per-flow predictions.

    Scores are model-specific ranking signals, NOT calibrated probabilities.
    """
    if features_df.empty:
        return []

    model_file = _model_path(settings)
    if retrain or not model_file.exists():
        meta = train_model(features_df, settings)
        mode = meta["mode"]
    else:
        import json

        meta_path = model_file.parent / "model_meta.json"
        mode = json.loads(meta_path.read_text(encoding="utf-8")).get("mode", "isolation_forest")

    artifact = joblib.load(model_file)
    model = artifact["model"]
    scaler = artifact["scaler"]
    cols = artifact.get("columns") or [c for c in NUMERIC_FEATURE_COLUMNS if c in features_df.columns]

    # Align columns
    frame = features_df.copy()
    for c in cols:
        if c not in frame.columns:
            frame[c] = 0.0
    X, _ = build_matrix(frame[cols] if set(cols).issubset(frame.columns) else frame)
    # Rebuild with exact training columns
    mat = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if "tls_present" in mat.columns:
        mat["tls_present"] = mat["tls_present"].astype(float)
    Xs = scaler.transform(mat.to_numpy(dtype=float))

    threshold = float(settings.get("ml", {}).get("suspicious_score_threshold", 0.55))
    results: list[dict[str, Any]] = []

    if mode == "isolation_forest" or model.__class__.__name__ == "IsolationForest":
        # decision_function: higher = more normal; invert to anomaly score in [0,1]-ish rank
        raw = model.decision_function(Xs)
        # Map so higher means more anomalous
        anomaly = -raw
        # Min-max within batch for dashboard ranking (still not calibrated)
        amin, amax = float(anomaly.min()), float(anomaly.max())
        if amax - amin < 1e-9:
            scores = np.zeros_like(anomaly)
        else:
            scores = (anomaly - amin) / (amax - amin)
        preds = model.predict(Xs)  # -1 anomaly, 1 normal
        for i, flow_id in enumerate(features_df["flow_id"].tolist()):
            score = float(scores[i])
            label = "suspicious" if preds[i] == -1 or score >= threshold else "benign"
            results.append(
                {
                    "flow_id": flow_id,
                    "label": label,
                    "score": score,
                    "model_mode": "isolation_forest",
                    "score_note": "batch-normalized anomaly rank; not a calibrated probability",
                }
            )
    else:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(Xs)[:, 1]
        else:
            scores = model.predict(Xs).astype(float)
        labels = model.predict(Xs)
        for i, flow_id in enumerate(features_df["flow_id"].tolist()):
            score = float(scores[i])
            pred = int(labels[i])
            label = "suspicious" if pred == 1 or score >= threshold else "benign"
            results.append(
                {
                    "flow_id": flow_id,
                    "label": label,
                    "score": score,
                    "model_mode": "random_forest",
                    "score_note": "classifier score / vote fraction; not a calibrated probability",
                }
            )

    logger.info(
        "Detection complete: mode=%s suspicious=%s/%s",
        results[0]["model_mode"] if results else "n/a",
        sum(1 for r in results if r["label"] == "suspicious"),
        len(results),
    )
    return results
