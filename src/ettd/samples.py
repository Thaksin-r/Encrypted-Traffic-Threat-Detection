"""Generate a tiny synthetic PCAP for offline demos and tests."""

from __future__ import annotations

from pathlib import Path

from scapy.all import IP, TCP, UDP, wrpcap  # type: ignore[attr-defined]


def write_sample_pcap(path: str | Path) -> Path:
    """Write a minimal multi-flow PCAP (synthetic lab traffic only)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    base = 1_700_000_000.0
    packets = []

    # Flow 1: short TCP exchange
    for i, (sport, dport, payload_len) in enumerate(
        [(40001, 80, 60), (80, 40001, 120), (40001, 80, 40), (80, 40001, 200)]
    ):
        pkt = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=sport, dport=dport, flags="PA") / (b"x" * payload_len)
        pkt.time = base + i * 0.01
        # Fix directions for reply packets
        if sport == 80:
            pkt = IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=40001, flags="PA") / (b"y" * payload_len)
            pkt.time = base + i * 0.01
        packets.append(pkt)

    # Flow 2: UDP DNS-like
    for i in range(3):
        pkt = IP(src="10.0.0.1", dst="10.0.0.53") / UDP(sport=53000, dport=53) / (b"q" * 40)
        pkt.time = base + 1.0 + i * 0.02
        packets.append(pkt)
        reply = IP(src="10.0.0.53", dst="10.0.0.1") / UDP(sport=53, dport=53000) / (b"r" * 80)
        reply.time = base + 1.0 + i * 0.02 + 0.005
        packets.append(reply)

    # Flow 3: bursty many small packets (rule-friendly)
    for i in range(50):
        pkt = IP(src="10.0.0.9", dst="10.0.0.8") / TCP(sport=50000, dport=443, flags="PA") / (b"z" * 20)
        pkt.time = base + 2.0 + i * 0.001
        packets.append(pkt)

    # Repeated connections to same dest (connection frequency)
    for n in range(10):
        pkt = IP(src="10.0.0.1", dst="203.0.113.50") / TCP(sport=41000 + n, dport=443, flags="S")
        pkt.time = base + 3.0 + n * 0.1
        packets.append(pkt)

    wrpcap(str(out), packets)
    return out
