"""Generate the synthetic sample PCAP used for demos and tests."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ettd.samples import write_sample_pcap

if __name__ == "__main__":
    out = ROOT / "data" / "samples" / "sample.pcap"
    path = write_sample_pcap(out)
    print(f"Wrote {path}")
