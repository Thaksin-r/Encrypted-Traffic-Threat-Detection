"""Bidirectional flow construction from packet metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ettd.ingestion.pcap_reader import PacketRecord


@dataclass
class Flow:
    flow_key: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_time: float
    end_time: float
    packets: list[PacketRecord] = field(default_factory=list)
    forward_lengths: list[int] = field(default_factory=list)
    backward_lengths: list[int] = field(default_factory=list)
    forward_times: list[float] = field(default_factory=list)
    backward_times: list[float] = field(default_factory=list)
    tls_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def fwd_packets(self) -> int:
        return len(self.forward_lengths)

    @property
    def bwd_packets(self) -> int:
        return len(self.backward_lengths)

    @property
    def fwd_bytes(self) -> int:
        return int(sum(self.forward_lengths))

    @property
    def bwd_bytes(self) -> int:
        return int(sum(self.backward_lengths))


def _canonical_key(
    src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: str
) -> str:
    """Order endpoints so A->B and B->A map to the same bidirectional flow key."""
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a <= b:
        return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
    return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"


def _is_syn_no_ack(pkt: PacketRecord) -> bool:
    flags = getattr(pkt, "tcp_flags", "") or ""
    if isinstance(flags, str):
        return "S" in flags and "A" not in flags
    try:
        val = int(flags)
        return (val & 0x02 != 0) and (val & 0x10 == 0)
    except (ValueError, TypeError):
        return False


def _finalize_flow(pkts: list[PacketRecord], key: str, proto: str) -> Flow:
    syn_pkt = next((p for p in pkts if _is_syn_no_ack(p)), None)
    if syn_pkt is not None:
        fwd_src_ip, fwd_dst_ip = syn_pkt.src_ip, syn_pkt.dst_ip
        fwd_src_port, fwd_dst_port = syn_pkt.src_port, syn_pkt.dst_port
    else:
        first = pkts[0]
        fwd_src_ip, fwd_dst_ip = first.src_ip, first.dst_ip
        fwd_src_port, fwd_dst_port = first.src_port, first.dst_port

    flow = Flow(
        flow_key=key,
        src_ip=fwd_src_ip,
        dst_ip=fwd_dst_ip,
        src_port=fwd_src_port,
        dst_port=fwd_dst_port,
        protocol=proto,
        start_time=pkts[0].timestamp,
        end_time=pkts[-1].timestamp,
        packets=list(pkts),
    )

    for pkt in pkts:
        is_forward = (pkt.src_ip == fwd_src_ip and pkt.src_port == fwd_src_port)
        if is_forward:
            flow.forward_lengths.append(pkt.length)
            flow.forward_times.append(pkt.timestamp)
        else:
            flow.backward_lengths.append(pkt.length)
            flow.backward_times.append(pkt.timestamp)

        if pkt.tls_meta:
            # Keep first non-empty handshake observations; never invent values.
            for k, v in pkt.tls_meta.items():
                if k not in flow.tls_meta or flow.tls_meta[k] in (None, "", []):
                    flow.tls_meta[k] = v

    return flow


def build_flows(
    packets: list[PacketRecord],
    idle_timeout_seconds: float = 120.0,
) -> list[Flow]:
    """Group packets into bidirectional flows with idle timeout splitting."""
    if not packets:
        return []

    active_packets: dict[str, list[PacketRecord]] = {}
    active_last_ts: dict[str, float] = {}
    active_proto: dict[str, str] = {}
    completed: list[Flow] = []

    ordered = sorted(packets, key=lambda p: p.timestamp)

    for pkt in ordered:
        key = _canonical_key(
            pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol
        )

        last_ts = active_last_ts.get(key)
        if last_ts is not None and (pkt.timestamp - last_ts) > idle_timeout_seconds:
            completed.append(_finalize_flow(active_packets[key], key, active_proto[key]))
            del active_packets[key]
            del active_last_ts[key]
            del active_proto[key]

        if key not in active_packets:
            active_packets[key] = []
            active_proto[key] = pkt.protocol

        active_packets[key].append(pkt)
        active_last_ts[key] = pkt.timestamp

    for key, pkts in active_packets.items():
        completed.append(_finalize_flow(pkts, key, active_proto[key]))

    completed.sort(key=lambda f: f.start_time)
    return completed
