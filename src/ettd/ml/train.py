"""Train Isolation Forest or Random Forest models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from ettd.ml.features_matrix import build_matrix, group_ids
from ettd.ml.mode_select import select_model_mode

logger = logging.getLogger(__name__)


def train_model(
    features_df: pd.DataFrame,
    settings: dict[str, Any],
    *,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train according to selected mode; persist model artifacts."""
    mode, labels_df, reason = select_model_mode(settings)
    ml_cfg = settings.get("ml", {})
    out_dir = Path(model_dir or ml_cfg.get("model_dir", "data/models"))
    if not out_dir.is_absolute():
        from ettd.config import PROJECT_ROOT

        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X, cols = build_matrix(features_df)
    meta: dict[str, Any] = {
        "mode": mode,
        "reason": reason,
        "feature_columns": cols,
        "n_samples": int(len(features_df)),
        "score_is_calibrated_probability": False,
    }

    if mode == "isolation_forest":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = IsolationForest(
            contamination=float(ml_cfg.get("contamination", 0.1)),
            random_state=int(ml_cfg.get("random_state", 42)),
            n_estimators=100,
        )
        model.fit(Xs)
        joblib.dump({"model": model, "scaler": scaler, "columns": cols}, out_dir / "model.joblib")
        meta["algorithm"] = "IsolationForest"
    else:
        assert labels_df is not None
        merged = features_df.merge(labels_df[["flow_id", "label"]], on="flow_id", how="inner")
        if merged.empty:
            raise ValueError("No overlap between features flow_id and labels flow_id")
        X_m, cols = build_matrix(merged)
        y = merged["label"].to_numpy(dtype=int)
        groups = group_ids(merged)

        n_groups = int(groups.nunique())
        if n_groups >= 3 and len(merged) >= 4:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=0.3,
                random_state=int(ml_cfg.get("random_state", 42)),
            )
            train_idx, test_idx = next(splitter.split(X_m, y, groups))
        else:
            # Tiny demo datasets: hold out last row only (still fit scaler on train)
            train_idx = np.arange(max(1, len(merged) - 1))
            test_idx = np.array([len(merged) - 1])

        # Fit scaler on train only to avoid leakage
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_m[train_idx])
        X_test = scaler.transform(X_m[test_idx])

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=int(ml_cfg.get("random_state", 42)),
            class_weight="balanced",
        )
        model.fit(X_train, y[train_idx])
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "columns": cols,
                "train_flow_ids": merged.iloc[train_idx]["flow_id"].tolist(),
                "test_flow_ids": merged.iloc[test_idx]["flow_id"].tolist(),
            },
            out_dir / "model.joblib",
        )
        meta["algorithm"] = "RandomForestClassifier"
        meta["train_size"] = int(len(train_idx))
        meta["test_size"] = int(len(test_idx))
        meta["y_test"] = y[test_idx].tolist()
        proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(X_test)
        )
        meta["y_score_test"] = np.asarray(proba, dtype=float).tolist()

    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Trained %s (%s)", mode, reason)
    return meta
