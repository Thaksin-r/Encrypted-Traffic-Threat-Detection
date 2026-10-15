"""End-to-end offline pipeline: extract -> detect -> investigate (offline tools)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ettd.agent.investigator import investigate_flow
from ettd.config import db_path, load_settings
from ettd.db.repository import Repository
from ettd.ml.infer import detect_suspicious
from ettd.pipeline_extract import extract_from_pcap
from ettd.samples import write_sample_pcap


def main() -> int:
    parser = argparse.ArgumentParser(description="ETTD end-to-end offline demo")
    parser.add_argument("--pcap", default=None, help="PCAP path (default: generate sample)")
    parser.add_argument("--use-llm", action="store_true", help="Require OPENAI_API_KEY")
    args = parser.parse_args()

    settings = load_settings()
    if args.pcap:
        pcap = Path(args.pcap)
    else:
        pcap = ROOT / "data" / "samples" / "sample.pcap"
        write_sample_pcap(pcap)

    cid, df = extract_from_pcap(
        pcap,
        out_csv=ROOT / "data" / "processed" / "flows.csv",
        persist_db=True,
        settings=settings,
    )
    print(f"capture_id={cid} flows={len(df)}")

    repo = Repository(db_path(settings))
    preds = detect_suspicious(df, settings, retrain=True)
    repo.save_predictions(cid, preds)
    suspicious = [p for p in preds if p["label"] == "suspicious"]
    print(f"suspicious={len(suspicious)}")

    target = suspicious[0]["flow_id"] if suspicious else df.iloc[0]["flow_id"]
    result = investigate_flow(target, repo, settings, use_llm=args.use_llm)
    print(json.dumps({"investigation_id": result.get("investigation_id"), "status": result.get("status")}, indent=2))
    out = ROOT / "data" / "processed" / "last_investigation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if result.get("status") in {"completed", "failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
