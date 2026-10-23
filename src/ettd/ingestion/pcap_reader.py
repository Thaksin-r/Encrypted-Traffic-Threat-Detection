"""Offline PCAP ingestion — metadata only, never decrypt payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scapy.all import IP, IPv6, TCP, UDP, load_layer, rdpcap  # type: ignore[attr-defined]

try:
    load_layer("tls")
    from scapy.layers.tls.handshake import TLSClientHello  # type: ignore
    from scapy.layers.tls.record import TLS  # type: ignore
except Exception:
    TLS = None  # type: ignore
    TLSClientHello = None  # type: ignore


@dataclass
class PacketRecord:
    """Observable packet metadata used for flow construction."""

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    length: int
    tls_meta: dict[str, Any] = field(default_factory=dict)
    tcp_flags: str = ""


def _extract_tls_handshake_meta(pkt: Any) -> dict[str, Any]:
    """
    Best-effort TLS *handshake* metadata from cleartext ClientHello fields.

    Does not decrypt application data. Missing fields are omitted here;
    feature extraction maps them to explicit unavailable/null values.
    """
    meta: dict[str, Any] = {}
    if TLS is None:
        return meta

    tls_layer = None
    if pkt.haslayer(TLS):
        tls_layer = pkt.getlayer(TLS)
    elif pkt.haslayer(TCP):
        raw_payload = bytes(pkt[TCP].payload)
        if len(raw_payload) >= 2 and raw_payload[0] == 0x16 and raw_payload[1] == 0x03:
            try:
                tls_layer = TLS(raw_payload)
            except Exception:
                tls_layer = None

    if tls_layer is None:
        return meta

    meta["tls_present"] = True
    ch = tls_layer.getlayer(TLSClientHello)
    if ch is None:
        return meta

    # Version / cipher suites when exposed by Scapy
    if hasattr(ch, "version") and ch.version is not None:
        meta["tls_version"] = int(ch.version)
    if hasattr(ch, "ciphers") and ch.ciphers is not None:
        try:
            meta["tls_cipher_suites"] = [int(c) for c in list(ch.ciphers)[:32]]
        except Exception:
            meta["tls_cipher_suites"] = None

    # SNI / ALPN via extensions when present
    exts = getattr(ch, "ext", None) or []
    for ext in exts:
        name = ext.__class__.__name__
        if "ServerName" in name or name == "TLS_Ext_ServerName":
            try:
                servernames = getattr(ext, "servernames", None) or []
                if servernames:
                    sn = servernames[0]
                    sni = getattr(sn, "servername", None) or bytes(sn).decode("utf-8", errors="replace")
                    if isinstance(sni, bytes):
                        sni = sni.decode("utf-8", errors="replace")
                    meta["tls_sni"] = str(sni)[:253]
            except Exception:
                meta["tls_sni"] = None
        if "ALPN" in name or "ALPN" in name.upper():
            try:
                protocols = getattr(ext, "protocol_name_list", None) or getattr(ext, "protocols", None)
                if protocols:
                    meta["tls_alpn"] = [str(p)[:64] for p in list(protocols)[:8]]
            except Exception:
                meta["tls_alpn"] = None
    return meta


def read_pcap(pcap_path: str | Path) -> list[PacketRecord]:
    """Read an offline PCAP and return packet metadata records."""
    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP not found: {path}")

    packets = rdpcap(str(path))
    records: list[PacketRecord] = []

    for pkt in packets:
        ts = float(getattr(pkt, "time", 0.0))
        ip_layer = None
        if pkt.haslayer(IP):
            ip_layer = pkt[IP]
            src_ip, dst_ip = ip_layer.src, ip_layer.dst
        elif pkt.haslayer(IPv6):
            ip_layer = pkt[IPv6]
            src_ip, dst_ip = ip_layer.src, ip_layer.dst
        else:
            continue

        tcp_flags = ""
        if pkt.haslayer(TCP):
            l4 = pkt[TCP]
            protocol = "TCP"
            tcp_flags = str(l4.flags)
        elif pkt.haslayer(UDP):
            l4 = pkt[UDP]
            protocol = "UDP"
        else:
            continue

        length = int(len(pkt))
        tls_meta = _extract_tls_handshake_meta(pkt) if protocol == "TCP" else {}

        records.append(
            PacketRecord(
                timestamp=ts,
                src_ip=str(src_ip),
                dst_ip=str(dst_ip),
                src_port=int(l4.sport),
                dst_port=int(l4.dport),
                protocol=protocol,
                length=length,
                tls_meta=tls_meta,
                tcp_flags=tcp_flags,
            )
        )
    return records
