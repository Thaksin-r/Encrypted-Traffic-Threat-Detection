from pathlib import Path

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


def test_llm_investigate_failure_sets_status_failed(sample_pcap, tmp_path, repo, monkeypatch):
    import ettd.agent.investigator as inv_mod

    repo2, settings, df = _seed(repo, sample_pcap, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-test")

    def _failing_llm(*args, **kwargs):
        raise RuntimeError("OpenAI connection timeout")

    monkeypatch.setattr(inv_mod, "_llm_investigate", _failing_llm)

    flow_id = df.iloc[0]["flow_id"]
    result = inv_mod.investigate_flow(flow_id, repo2, settings, use_llm=True)
    assert result["status"] == "failed"
    assert "OpenAI connection timeout" in result["error"]

    # Check database status was updated and did not get stuck on 'pending'
    db_record = repo2.get_investigation(result["investigation_id"])
    assert db_record is not None
    assert db_record["status"] == "failed"


def test_ja3_blocklist_threat_intel_match(sample_pcap, tmp_path, repo):
    repo2, settings, _ = _seed(repo, sample_pcap, tmp_path)
    tools = InvestigationTools(repo2, settings)

    # 1. Indicator matching config/ja3_blocklist.txt
    res = tools.check_threat_intelligence(["bca193bf3b6d2156cfbe0e6b4b306d3e"])
    assert res["lookup_performed"] is True
    assert len(res["matches"]) == 1
    assert res["matches"][0]["type"] == "ja3"

    # 2. Benign / unknown indicator
    res_clean = tools.check_threat_intelligence(["00000000000000000000000000000000"])
    assert res_clean["lookup_performed"] is True
    assert len(res_clean["matches"]) == 0


def test_beaconing_detection_rule(tmp_path, repo):
    # Seed repo with beaconing flows
    from ettd.features.extractor import flows_to_dataframe
    from ettd.flows.builder import build_flows
    from ettd.ingestion.pcap_reader import PacketRecord

    packets = []
    base = 1700000000.0
    for i in range(6):
        t = base + i * 10.0
        packets.append(PacketRecord(t, "192.168.1.100", "198.51.100.2", 40000 + i, 443, "TCP", 60))

    flows = build_flows(packets, idle_timeout_seconds=5.0)
    df = flows_to_dataframe(flows)
    cid = repo.create_capture("beacon_test.pcap")
    repo.save_flows_and_features(cid, df)

    settings = {
        "detection_rules_path": str(
            Path(__file__).resolve().parents[1] / "config" / "detection_rules.yaml"
        ),
        "threat_intelligence": {"source": None},
    }
    tools = InvestigationTools(repo, settings)
    rule_res = tools.compare_detection_rules(df.iloc[0]["flow_id"])
    assert "beaconing_regular_timing" in rule_res["rules_checked"]
    matched_ids = [m["rule_id"] for m in rule_res["matched_rules"]]
    assert "beaconing_regular_timing" in matched_ids


