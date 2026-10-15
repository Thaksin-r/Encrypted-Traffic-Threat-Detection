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
) -> tuple[str, str, str, int, int, str]:
    """Order endpoints so A->B and B->A map to the same bidirectional flow."""
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a <= b:
        return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}", src_ip, dst_ip, src_port, dst_port, protocol
    return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}", dst_ip, src_ip, dst_port, src_port, protocol


def build_flows(
    packets: list[PacketRecord],
    idle_timeout_seconds: float = 120.0,
) -> list[Flow]:
    """Group packets into bidirectional flows with idle timeout splitting."""
    active: dict[str, Flow] = {}
    completed: list[Flow] = []

    ordered = sorted(packets, key=lambda p: p.timestamp)

    for pkt in ordered:
        key, c_src, c_dst, c_sport, c_dport, proto = _canonical_key(
            pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol
        )

        flow = active.get(key)
        if flow is not None and (pkt.timestamp - flow.end_time) > idle_timeout_seconds:
            completed.append(flow)
            del active[key]
            flow = None

        if flow is None:
            flow = Flow(
                flow_key=key,
                src_ip=c_src,
                dst_ip=c_dst,
                src_port=c_sport,
                dst_port=c_dport,
                protocol=proto,
                start_time=pkt.timestamp,
                end_time=pkt.timestamp,
            )
            active[key] = flow

        flow.end_time = pkt.timestamp
        flow.packets.append(pkt)

        is_forward = pkt.src_ip == flow.src_ip and pkt.src_port == flow.src_port
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

    completed.extend(active.values())
    completed.sort(key=lambda f: f.start_time)
    return completed
