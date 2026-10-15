"""CLI: PCAP -> flows -> features (CSV and/or SQLite)."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from ettd.config import db_path, load_settings
from ettd.db.repository import Repository
from ettd.features.extractor import flows_to_dataframe
from ettd.flows.builder import build_flows
from ettd.ingestion.pcap_reader import read_pcap


def extract_from_pcap(
    pcap_path: str | Path,
    *,
    out_csv: str | Path | None = None,
    persist_db: bool = True,
    capture_id: str | None = None,
    settings: dict | None = None,
):
    settings = settings or load_settings()
    path = Path(pcap_path)
    packets = read_pcap(path)
    idle = float(settings.get("flow_idle_timeout_seconds", 120))
    flows = build_flows(packets, idle_timeout_seconds=idle)

    cid = capture_id or str(uuid.uuid4())
    df = flows_to_dataframe(flows, flow_id_prefix=f"c{cid[:8]}")
    flow_times = {
        f"{df.iloc[i]['flow_id']}": (flows[i].start_time, flows[i].end_time)
        for i in range(len(flows))
    }

    if out_csv:
        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    if persist_db:
        repo = Repository(db_path(settings))
        repo.create_capture(str(path.resolve()), capture_id=cid)
        repo.save_flows_and_features(cid, df, flow_times=flow_times)

    return cid, df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract flow features from an offline PCAP (no decryption)."
    )
    parser.add_argument("pcap", type=str, help="Path to authorized offline PCAP")
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/flows.csv",
        help="Output CSV path",
    )
    parser.add_argument("--no-db", action="store_true", help="Skip SQLite persistence")
    parser.add_argument("--capture-id", type=str, default=None)
    args = parser.parse_args(argv)

    cid, df = extract_from_pcap(
        args.pcap,
        out_csv=args.out,
        persist_db=not args.no_db,
        capture_id=args.capture_id,
    )
    print(f"capture_id={cid}")
    print(f"flows={len(df)}")
    print(f"csv={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
