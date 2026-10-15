from pathlib import Path

import pandas as pd

from ettd.ml.evaluate import evaluate_binary, unsupervised_summary
from ettd.ml.features_matrix import group_ids
from ettd.ml.infer import detect_suspicious
from ettd.ml.mode_select import select_model_mode, validate_labels_csv
from ettd.pipeline_extract import extract_from_pcap


def test_mode_select_isolation_by_default():
    settings = {"ml": {"mode": "auto"}, "labels": {"path": None}}
    mode, labels, reason = select_model_mode(settings)
    assert mode == "isolation_forest"
    assert labels is None


def test_mode_select_random_forest_with_labels(tmp_path, sample_pcap):
    settings = {
        "ml": {"mode": "auto", "model_dir": str(tmp_path / "models"), "random_state": 0, "contamination": 0.2},
        "labels": {"path": None},
        "database": {"path": str(tmp_path / "db.sqlite")},
    }
    _, df = extract_from_pcap(sample_pcap, persist_db=False, settings=settings)
    labels_path = tmp_path / "labels.csv"
    labels = pd.DataFrame(
        {
            "flow_id": df["flow_id"].tolist(),
            "label": ["suspicious" if i % 3 == 0 else "benign" for i in range(len(df))],
        }
    )
    labels.to_csv(labels_path, index=False)
    settings["labels"]["path"] = str(labels_path)
    ok, reason, labels_df = validate_labels_csv(labels_path)
    assert ok, reason
    mode, labels_df2, _ = select_model_mode(settings)
    assert mode == "random_forest"
    assert labels_df2 is not None

    preds = detect_suspicious(df, settings, retrain=True)
    assert len(preds) == len(df)
    assert preds[0]["model_mode"] == "random_forest"
    assert "calibrated" not in preds[0]["score_note"] or "not a calibrated" in preds[0]["score_note"]


def test_isolation_forest_detect(sample_pcap, tmp_path):
    settings = {
        "ml": {
            "mode": "isolation_forest",
            "model_dir": str(tmp_path / "models"),
            "random_state": 0,
            "contamination": 0.3,
            "suspicious_score_threshold": 0.55,
        },
        "labels": {"path": None},
        "database": {"path": str(tmp_path / "db.sqlite")},
    }
    _, df = extract_from_pcap(sample_pcap, persist_db=False, settings=settings)
    preds = detect_suspicious(df, settings, retrain=True)
    assert len(preds) == len(df)
    assert all(p["model_mode"] == "isolation_forest" for p in preds)
    summary = unsupervised_summary([p["score"] for p in preds])
    assert summary["n"] == len(preds)


def test_group_ids_stable():
    df = pd.DataFrame(
        {
            "flow_id": ["a", "b"],
            "flow_key": ["k1", "k1"],
            "src_ip": ["1", "1"],
            "dst_ip": ["2", "2"],
        }
    )
    groups = group_ids(df)
    assert groups.tolist() == ["k1", "k1"]


def test_evaluate_binary():
    metrics = evaluate_binary([0, 0, 1, 1], [0, 1, 1, 1])
    assert "precision" in metrics
    assert "false_positive_rate" in metrics
    assert metrics["confusion_matrix"]["fp"] == 1
