"""Generate synthetic PCAPs for offline demos and tests."""

from __future__ import annotations

from pathlib import Path

from scapy.all import IP, TCP, UDP, load_layer, wrpcap  # type: ignore[attr-defined]

load_layer("tls")
from scapy.layers.tls.extensions import ServerName, TLS_Ext_ServerName  # type: ignore
from scapy.layers.tls.handshake import TLSClientHello  # type: ignore
from scapy.layers.tls.record import TLS  # type: ignore


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


def write_rich_sample_pcap(path: str | Path) -> Path:
    """
    Write a rich synthetic PCAP containing:
    - Real TLS ClientHellos with SNI and cipher suites on ports 443 and 8443
    - A periodic beaconing host (regular timing interval)
    - Benign web browsing flows across distinct hosts
    - Exfiltration / high-asymmetry traffic
    - Short packet burst traffic
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    base = 1_700_000_000.0
    packets = []

    # 1. Normal TLS browsing on port 443 (api.github.com)
    for i in range(10):
        p_req = (
            IP(src="10.0.1.10", dst="140.82.121.4")
            / TCP(sport=52100 + i, dport=443, flags="PA")
            / TLS(
                msg=[
                    TLSClientHello(
                        version=0x0303,
                        ciphers=[0x1301, 0x1302, 0xC02F, 0xC030],
                        ext=[TLS_Ext_ServerName(servernames=[ServerName(servername=b"api.github.com")])],
                    )
                ]
            )
        )
        p_req.time = base + i * 1.5
        packets.append(p_req)

        p_resp = (
            IP(src="140.82.121.4", dst="10.0.1.10")
            / TCP(sport=443, dport=52100 + i, flags="PA")
            / (b"\x16\x03\x03\x00\x40" + b"\x02" + b"s" * 60)
        )
        p_resp.time = base + i * 1.5 + 0.05
        packets.append(p_resp)

    # 2. TLS on non-standard port 8443 (internal-vault.corp)
    for i in range(5):
        p_tls8443 = (
            IP(src="10.0.1.15", dst="192.168.10.50")
            / TCP(sport=53100 + i, dport=8443, flags="PA")
            / TLS(
                msg=[
                    TLSClientHello(
                        version=0x0303,
                        ciphers=[0xC02B, 0xC02F],
                        ext=[TLS_Ext_ServerName(servernames=[ServerName(servername=b"internal-vault.corp")])],
                    )
                ]
            )
        )
        p_tls8443.time = base + 30.0 + i * 2.0
        packets.append(p_tls8443)

    # 3. Beaconing Host: 10.0.2.88 -> 198.51.100.99 (C2 Server)
    # 8 distinct flows spaced exactly 150 seconds apart (idle timeout > 120s splits them)
    # Low CV, regular timing interval
    for n in range(8):
        flow_time = base + 100.0 + n * 150.0
        # TCP SYN
        p_syn = IP(src="10.0.2.88", dst="198.51.100.99") / TCP(sport=45000 + n, dport=443, flags="S")
        p_syn.time = flow_time
        packets.append(p_syn)
        # TCP SYN-ACK
        p_sa = IP(src="198.51.100.99", dst="10.0.2.88") / TCP(sport=443, dport=45000 + n, flags="SA")
        p_sa.time = flow_time + 0.02
        packets.append(p_sa)
        # ClientHello beacon
        p_beacon = (
            IP(src="10.0.2.88", dst="198.51.100.99")
            / TCP(sport=45000 + n, dport=443, flags="PA")
            / TLS(
                msg=[
                    TLSClientHello(
                        version=0x0303,
                        ciphers=[0x0035, 0x002F],
                        ext=[TLS_Ext_ServerName(servernames=[ServerName(servername=b"update-check.service")])],
                    )
                ]
            )
        )
        p_beacon.time = flow_time + 0.05
        packets.append(p_beacon)
        # Beacon reply
        p_ack = (
            IP(src="198.51.100.99", dst="10.0.2.88")
            / TCP(sport=443, dport=45000 + n, flags="PA")
            / (b"\x16\x03\x03\x00\x20" + b"b" * 32)
        )
        p_ack.time = flow_time + 0.08
        packets.append(p_ack)

    # 4. Exfiltration / High Asymmetry Flow (10.0.1.200 -> 203.0.113.88)
    for i in range(25):
        p_exfil = (
            IP(src="10.0.1.200", dst="203.0.113.88")
            / TCP(sport=56000, dport=443, flags="PA")
            / (b"d" * 500)
        )
        p_exfil.time = base + 50.0 + i * 0.02
        packets.append(p_exfil)
    p_exfil_ack = IP(src="203.0.113.88", dst="10.0.1.200") / TCP(sport=443, dport=56000, flags="A")
    p_exfil_ack.time = base + 51.0
    packets.append(p_exfil_ack)

    # 5. Short Burst of small packets (10.0.1.77 -> 10.0.1.88)
    for i in range(50):
        p_burst = IP(src="10.0.1.77", dst="10.0.1.88") / TCP(sport=57000, dport=443, flags="PA") / (b"p" * 20)
        p_burst.time = base + 80.0 + i * 0.002
        packets.append(p_burst)

    # 6. UDP DNS Traffic
    for i in range(6):
        p_dns = IP(src="10.0.1.10", dst="10.0.0.53") / UDP(sport=54000 + i, dport=53) / (b"dnsquery" * 4)
        p_dns.time = base + 5.0 + i * 3.0
        packets.append(p_dns)
        p_dns_rep = IP(src="10.0.0.53", dst="10.0.1.10") / UDP(sport=53, dport=54000 + i) / (b"dnsanswer" * 6)
        p_dns_rep.time = base + 5.0 + i * 3.0 + 0.01
        packets.append(p_dns_rep)

    # Sort all packets by timestamp
    packets.sort(key=lambda p: float(p.time))
    wrpcap(str(out), packets)
    return out
