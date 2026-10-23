"""Generate synthetic sample PCAPs used for demos, tests, and dashboards."""

from __future__ import annotations

import argparse
from pathlib import Path

from ettd.config import PROJECT_ROOT
from ettd.samples import write_rich_sample_pcap, write_sample_pcap


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic sample PCAP")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(PROJECT_ROOT / "data" / "samples" / "sample.pcap"),
        help="Destination PCAP file path",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Generate minimal test PCAP instead of rich multi-flow demo capture",
    )
    args = parser.parse_args()

    out = Path(args.output)
    if args.minimal:
        path = write_sample_pcap(out)
        print(f"Wrote minimal sample PCAP to {path}")
    else:
        path = write_rich_sample_pcap(out)
        print(f"Wrote rich demo PCAP (TLS ClientHellos + Beaconing host) to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
