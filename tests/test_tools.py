from ettd.agent.investigator import investigate_flow
from ettd.agent.tools import InvestigationTools
from ettd.config import load_settings
from ettd.ml.infer import detect_suspicious
from ettd.pipeline_extract import extract_from_pcap


def _seed(repo, sample_pcap, tmp_path):
    settings = load_settings()
    settings["database"]["path"] = str(tmp_path / "ettd.db")
    settings["ml"] = {
        "mode": "isolation_forest",
        "model_dir": str(tmp_path / "models"),
        "contamination": 0.3,
        "random_state": 0,
        "suspicious_score_threshold": 0.5,
    }
    settings["threat_intelligence"] = {"source": None}
    cid, df = extract_from_pcap(sample_pcap, persist_db=True, settings=settings)
    # Re-bind repo path
    from ettd.db.repository import Repository

    repo2 = Repository(tmp_path / "ettd.db")
    preds = detect_suspicious(df, settings, retrain=True)
    repo2.save_predictions(cid, preds)
    return repo2, settings, df


def test_tools_happy_path(sample_pcap, tmp_path, repo):
    repo2, settings, df = _seed(repo, sample_pcap, tmp_path)
    tools = InvestigationTools(repo2, settings)
    flow_id = df.iloc[0]["flow_id"]

    inspect = tools.inspect_flow(flow_id)
    assert "features" in inspect
    tls = tools.analyze_tls_metadata(flow_id)
    assert tls["tls_sni"] == "unavailable" or tls["tls_present"] in (True, False)
    related = tools.find_related_flows(flow_id)
    assert "related_flows" in related
    rules = tools.compare_detection_rules(flow_id)
    assert "rules_checked" in rules
    ti = tools.check_threat_intelligence(["203.0.113.50"])
    assert ti["lookup_performed"] is False
    assert "no threat intelligence source configured" in ti["reason"]


def test_invalid_flow_id(sample_pcap, tmp_path, repo):
    repo2, settings, _ = _seed(repo, sample_pcap, tmp_path)
    tools = InvestigationTools(repo2, settings)
    missing = tools.inspect_flow("flow_does_not_exist_000")
    assert missing.get("error") == "flow_not_found"


def test_investigate_offline_tools(sample_pcap, tmp_path, repo):
    repo2, settings, df = _seed(repo, sample_pcap, tmp_path)
    flow_id = df.iloc[0]["flow_id"]
    result = investigate_flow(flow_id, repo2, settings, use_llm=False)
    assert result["status"] == "completed"
    assert result["report"] is not None
    assert "incident_summary" in result["report"]
    assert result["tool_calls"]


def test_investigate_without_api_key_fails_clearly(sample_pcap, tmp_path, repo, monkeypatch):
    repo2, settings, df = _seed(repo, sample_pcap, tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = investigate_flow(df.iloc[0]["flow_id"], repo2, settings, use_llm=True)
    assert result["status"] == "failed"
    assert result["report"] is None
    assert "OPENAI_API_KEY" in result["error"]
