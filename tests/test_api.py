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
    monkeypatch.setenv("ETTD_ALLOWED_PCAP_DIRS", str(tmp_path))

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


def test_extract_path_traversal_rejected(client):
    r = client.post("/captures/extract", params={"pcap_path": "../../secret.pcap"})
    assert r.status_code == 403
    assert "forbidden" in r.json()["detail"].lower() or "traversal" in r.json()["detail"].lower()


def test_upload_unique_filename_and_size_limit(client, tmp_path):
    pcap = tmp_path / "u.pcap"
    write_sample_pcap(pcap)
    content = pcap.read_bytes()

    # Upload 1
    r1 = client.post("/captures/upload", files={"file": ("u.pcap", content, "application/vnd.tcpdump.pcap")})
    assert r1.status_code == 200, r1.text
    fn1 = r1.json()["filename"]

    # Upload 2 with same original name: must produce unique filename, no overwriting
    r2 = client.post("/captures/upload", files={"file": ("u.pcap", content, "application/vnd.tcpdump.pcap")})
    assert r2.status_code == 200, r2.text
    fn2 = r2.json()["filename"]
    assert fn1 != fn2

    # Check size limit rejects oversized file
    from unittest.mock import patch
    with patch("ettd.api.main.MAX_UPLOAD_SIZE", 100):
        r_large = client.post("/captures/upload", files={"file": ("large.pcap", b"x" * 200, "application/vnd.tcpdump.pcap")})
        assert r_large.status_code == 413


def test_flows_pagination_sorting_explain_and_stats(client, tmp_path):
    pcap = tmp_path / "sample_stat.pcap"
    write_sample_pcap(pcap)
    r = client.post("/captures/extract", params={"pcap_path": str(pcap)})
    assert r.status_code == 200

    r_train = client.post("/train", json={})
    assert r_train.status_code == 200

    r_detect = client.post("/detect", json={})
    assert r_detect.status_code == 200
    preds = r_detect.json()["predictions"]
    flow_id = preds[0]["flow_id"]

    # 1. Flows pagination & sorting
    r_flows = client.get("/flows", params={"limit": 2, "offset": 0, "sort_by": "score", "sort_order": "desc"})
    assert r_flows.status_code == 200
    flows = r_flows.json()
    assert len(flows) <= 2
    assert "features" in flows[0]
    assert "prediction" in flows[0]

    # 2. Score distribution
    r_dist = client.get("/stats/score-distribution")
    assert r_dist.status_code == 200
    dist = r_dist.json()
    assert "bin_edges" in dist
    assert "counts" in dist
    assert "threshold" in dist

    # 3. Timeline
    r_time = client.get("/stats/timeline")
    assert r_time.status_code == 200
    timeline = r_time.json()
    assert isinstance(timeline, list)
    if timeline:
        assert "time" in timeline[0]
        assert "flow_count" in timeline[0]

    # 4. Top talkers
    r_talkers = client.get("/stats/top-talkers")
    assert r_talkers.status_code == 200
    talkers = r_talkers.json()
    assert "top_destinations_by_flows" in talkers
    assert "top_destinations_by_bytes" in talkers
    assert "top_ports" in talkers

    # 5. Explain flow
    r_exp = client.get(f"/flows/{flow_id}/explain")
    assert r_exp.status_code == 200
    exp = r_exp.json()
    assert exp["flow_id"] == flow_id
    assert "top_features" in exp
    assert len(exp["top_features"]) <= 5
    for item in exp["top_features"]:
        assert "feature" in item
        assert "score" in item
        assert isinstance(item["score"], (int, float))

