"""Command-line entry points for ETTD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ettd.agent.investigator import investigate_flow
from ettd.config import PROJECT_ROOT, db_path, load_settings
from ettd.db.repository import Repository
from ettd.ml.evaluate import unsupervised_summary
from ettd.ml.infer import detect_suspicious
from ettd.ml.mode_select import select_model_mode
from ettd.pipeline_extract import extract_from_pcap, main as pipeline_extract_main
from ettd.samples import write_sample_pcap


def extract_main(argv: list[str] | None = None) -> int:
    """CLI entry point for ettd-extract."""
    return pipeline_extract_main(argv)


def detect_main(argv: list[str] | None = None) -> int:
    """CLI entry point for ettd-detect."""
    parser = argparse.ArgumentParser(description="Run dual-mode ML detection")
    parser.add_argument("--capture-id", default=None)
    parser.add_argument("--no-retrain", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings()
    mode, _, reason = select_model_mode(settings)
    print(f"selected_mode={mode} reason={reason}")

    repo = Repository(db_path(settings))
    df = repo.get_features_dataframe(args.capture_id)
    if df.empty:
        print("No features in database. Run extraction first.")
        return 1

    preds = detect_suspicious(df, settings, retrain=not args.no_retrain)
    if args.capture_id:
        repo.save_predictions(args.capture_id, preds)
    else:
        flow_cap = {f["flow_id"]: f["capture_id"] for f in repo.list_flows()}
        by_cap: dict[str, list] = {}
        for p in preds:
            cid = flow_cap.get(p["flow_id"])
            if cid:
                by_cap.setdefault(cid, []).append(p)
        for cid, items in by_cap.items():
            repo.save_predictions(cid, items)

    suspicious = [p for p in preds if p["label"] == "suspicious"]
    print(f"flows={len(preds)} suspicious={len(suspicious)} benign={len(preds) - len(suspicious)}")
    print(unsupervised_summary([p["score"] for p in preds]))
    for p in suspicious[:10]:
        print(f"  {p['flow_id']} score={p['score']:.4f} mode={p['model_mode']}")
    return 0


def pipeline_main(argv: list[str] | None = None) -> int:
    """CLI entry point for ettd-pipeline."""
    parser = argparse.ArgumentParser(description="ETTD end-to-end offline demo")
    parser.add_argument("--pcap", default=None, help="PCAP path (default: generate sample)")
    parser.add_argument("--use-llm", action="store_true", help="Require OPENAI_API_KEY")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.pcap:
        pcap = Path(args.pcap)
    else:
        pcap = PROJECT_ROOT / "data" / "samples" / "sample.pcap"
        write_sample_pcap(pcap)

    cid, df = extract_from_pcap(
        pcap,
        out_csv=PROJECT_ROOT / "data" / "processed" / "flows.csv",
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
    out = PROJECT_ROOT / "data" / "processed" / "last_investigation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if result.get("status") in {"completed", "failed"} else 1


if __name__ == "__main__":
    raise SystemExit(pipeline_main())
