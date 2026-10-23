from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ettd.api.main import app
from ettd.samples import write_sample_pcap


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = tmp_path / "api.db"
    models = tmp_path / "models"
    monkeypatch.setenv("ETTD_DB_PATH", str(db))

    def _settings(settings_path=None):
        return {
            "flow_idle_timeout_seconds": 120,
            "ml": {
                "mode": "isolation_forest",
                "contamination": 0.3,
                "random_state": 0,
                "model_dir": str(models),
                "suspicious_score_threshold": 0.5,
            },
            "labels": {"path": None},
            "threat_intelligence": {"source": None},
            "database": {"path": str(db)},
            "detection_rules_path": str(
                Path(__file__).resolve().parents[1] / "config" / "detection_rules.yaml"
            ),
        }

    import ettd.api.main as api_main
    import ettd.config as config
    import ettd.pipeline_extract as pipeline_extract

    monkeypatch.setattr(config, "load_settings", _settings)
    monkeypatch.setattr(api_main, "load_settings", _settings)
    monkeypatch.setattr(pipeline_extract, "load_settings", _settings)
    monkeypatch.setattr(api_main, "db_path", lambda settings=None: Path(db))
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_extract_detect_investigate_flow(client, tmp_path):
    pcap = tmp_path / "s.pcap"
    write_sample_pcap(pcap)
    r = client.post("/captures/extract", params={"pcap_path": str(pcap)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["flow_count"] >= 1

    # /detect without a trained model must fail clearly with 400
    r_err = client.post("/detect", json={})
    assert r_err.status_code == 400, r_err.text
    assert "No trained model found" in r_err.json()["detail"]

    # Train model
    r_train = client.post("/train", json={})
    assert r_train.status_code == 200, r_train.text
    assert r_train.json()["status"] == "trained"

    # Now /detect loads the trained model
    r = client.post("/detect", json={})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert preds

    flow_id = preds[0]["flow_id"]
    r = client.get(f"/flows/{flow_id}")
    assert r.status_code == 200

    r = client.post("/investigate", json={"flow_id": flow_id, "use_llm": False})
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


def test_extract_missing_pcap(client):
    r = client.post("/captures/extract", params={"pcap_path": "does_not_exist.pcap"})
    assert r.status_code == 404


def test_investigate_unknown_flow(client):
    r = client.post("/investigate", json={"flow_id": "missing_flow_id", "use_llm": False})
    assert r.status_code == 404
