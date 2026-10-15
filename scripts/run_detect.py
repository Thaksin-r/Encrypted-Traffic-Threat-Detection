"""Train/infer ML detection on features in SQLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ettd.config import db_path, load_settings
from ettd.db.repository import Repository
from ettd.ml.evaluate import unsupervised_summary
from ettd.ml.infer import detect_suspicious
from ettd.ml.mode_select import select_model_mode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dual-mode ML detection")
    parser.add_argument("--capture-id", default=None)
    parser.add_argument("--no-retrain", action="store_true")
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
