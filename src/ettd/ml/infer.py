"""Run detection inference on flow features."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ettd.ml.features_matrix import NUMERIC_FEATURE_COLUMNS
from ettd.ml.train import train_model

logger = logging.getLogger(__name__)


def _resolve_model_and_meta(settings: dict[str, Any]) -> tuple[Path, Path]:
    from ettd.config import PROJECT_ROOT

    model_dir = Path(settings.get("ml", {}).get("model_dir", "data/models"))
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir

    latest_json = model_dir / "latest.json"
    if latest_json.exists():
        try:
            info = json.loads(latest_json.read_text(encoding="utf-8"))
            m_path = Path(info.get("model_file", ""))
            meta_path = Path(info.get("meta_file", ""))
            if m_path.exists():
                return m_path, meta_path
        except Exception:
            pass

    return model_dir / "model.joblib", model_dir / "model_meta.json"


def detect_suspicious(
    features_df: pd.DataFrame,
    settings: dict[str, Any],
    *,
    retrain: bool = False,
) -> list[dict[str, Any]]:
    """
    Return per-flow predictions.

    Scores are model-specific ranking signals, NOT calibrated probabilities.
    """
    if features_df.empty:
        return []

    model_file, meta_path = _resolve_model_and_meta(settings)
    if retrain:
        meta = train_model(features_df, settings)
        model_file, meta_path = _resolve_model_and_meta(settings)
        mode = meta.get("mode", "isolation_forest")
    elif not model_file.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_file}. Train a model first using POST /train."
        )
    else:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mode = meta.get("mode", "isolation_forest")
        else:
            meta = {}
            mode = "isolation_forest"

    artifact = joblib.load(model_file)
    model = artifact["model"]
    scaler = artifact["scaler"]
    cols = artifact.get("columns") or [c for c in NUMERIC_FEATURE_COLUMNS if c in features_df.columns]

    # Align columns
    frame = features_df.copy()
    for c in cols:
        if c not in frame.columns:
            frame[c] = 0.0

    mat = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if "tls_present" in mat.columns:
        mat["tls_present"] = mat["tls_present"].astype(float)
    if "tls_cipher_suites_missing" in mat.columns:
        mat["tls_cipher_suites_missing"] = mat["tls_cipher_suites_missing"].astype(float)
    Xs = scaler.transform(mat.to_numpy(dtype=float))

    results: list[dict[str, Any]] = []

    if mode == "isolation_forest" or model.__class__.__name__ == "IsolationForest":
        # decision_function: higher = more normal; < 0 is anomalous
        raw = model.decision_function(Xs)
        threshold = float(meta.get("threshold", 0.0))
        if hasattr(model, "score_samples"):
            scores = -model.score_samples(Xs)
        else:
            scores = -raw
        preds = model.predict(Xs)  # -1 anomaly, 1 normal
        for i, flow_id in enumerate(features_df["flow_id"].tolist()):
            score = float(scores[i])
            label = "suspicious" if (preds[i] == -1 or raw[i] < threshold) else "benign"
            results.append(
                {
                    "flow_id": flow_id,
                    "label": label,
                    "score": score,
                    "model_mode": "isolation_forest",
                    "score_note": "anomaly score (unsupervised); not a calibrated probability",
                }
            )
    elif mode == "autoencoder" or model.__class__.__name__ == "MLPRegressor":
        reconstructed = model.predict(Xs)
        recon_errors = np.mean(np.square(Xs - reconstructed), axis=1)
        dynamic_thresh = float(meta.get("dynamic_threshold", meta.get("threshold", 0.5)))

        for i, flow_id in enumerate(features_df["flow_id"].tolist()):
            err = float(recon_errors[i])
            # Calibrated normalized anomaly index: 0.5 at dynamic threshold
            norm_score = float(err / (err + dynamic_thresh + 1e-9) * 2.0)
            norm_score = float(min(1.0, max(0.0, norm_score)))
            label = "suspicious" if err >= dynamic_thresh else "benign"
            results.append(
                {
                    "flow_id": flow_id,
                    "label": label,
                    "score": norm_score,
                    "raw_reconstruction_error": err,
                    "model_mode": "autoencoder",
                    "score_note": f"Autoencoder reconstruction error: {err:.4f} (dynamic threshold: {dynamic_thresh:.4f})",
                }
            )
    else:
        threshold = float(meta.get("threshold", settings.get("ml", {}).get("suspicious_score_threshold", 0.55)))
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(Xs)[:, 1]
        else:
            scores = model.predict(Xs).astype(float)
        labels = model.predict(Xs)
        train_flow_ids = set(artifact.get("train_flow_ids", []))
        for i, flow_id in enumerate(features_df["flow_id"].tolist()):
            score = float(scores[i])
            pred = int(labels[i])
            label = "suspicious" if (pred == 1 or score >= threshold) else "benign"
            is_training_row = flow_id in train_flow_ids
            score_note = (
                "classifier score on training row (not an out-of-sample test evaluation)"
                if is_training_row
                else "classifier score / vote fraction; not a calibrated probability"
            )
            results.append(
                {
                    "flow_id": flow_id,
                    "label": label,
                    "score": score,
                    "model_mode": "random_forest",
                    "is_training_row": is_training_row,
                    "score_note": score_note,
                }
            )

    logger.info(
        "Detection complete: mode=%s suspicious=%s/%s",
        results[0]["model_mode"] if results else "n/a",
        sum(1 for r in results if r["label"] == "suspicious"),
        len(results),
    )
    return results
