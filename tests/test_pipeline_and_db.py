from pathlib import Path

from ettd.config import load_settings
from ettd.db.repository import Repository
from ettd.pipeline_extract import extract_from_pcap


def test_extract_to_csv_and_db(sample_pcap, tmp_path):
    settings = load_settings()
    settings["database"]["path"] = str(tmp_path / "ettd.db")
    out_csv = tmp_path / "flows.csv"
    cid, df = extract_from_pcap(
        sample_pcap,
        out_csv=out_csv,
        persist_db=True,
        settings=settings,
    )
    assert len(df) >= 1
    assert out_csv.exists()
    repo = Repository(tmp_path / "ettd.db")
    captures = repo.list_captures()
    assert any(c["capture_id"] == cid for c in captures)
    assert len(repo.list_flows(cid)) == len(df)
    feat = repo.get_flow_features(df.iloc[0]["flow_id"])
    assert feat is not None
    assert "duration" in feat


def test_rerun_creates_versioned_capture(sample_pcap, tmp_path):
    settings = load_settings()
    settings["database"]["path"] = str(tmp_path / "ettd.db")
    cid1, _ = extract_from_pcap(sample_pcap, persist_db=True, settings=settings)
    cid2, _ = extract_from_pcap(sample_pcap, persist_db=True, settings=settings)
    assert cid1 != cid2
    repo = Repository(tmp_path / "ettd.db")
    assert len(repo.list_captures()) >= 2
