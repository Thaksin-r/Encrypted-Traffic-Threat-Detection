"""Flow-level feature extraction from metadata (no payload decryption)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ettd.flows.builder import Flow

FEATURE_COLUMNS = [
    "flow_id",
    "flow_key",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "duration",
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "total_packets",
    "total_bytes",
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_min",
    "pkt_len_max",
    "iat_mean",
    "iat_std",
    "iat_min",
    "iat_max",
    "bytes_per_second",
    "packets_per_second",
    "fwd_bwd_byte_ratio",
    "dest_connection_count",
    "src_connection_count",
    "tls_present",
    "tls_version",
    "tls_sni",
    "tls_alpn",
    "tls_cipher_suite_count",
    "tls_cipher_suites_missing",
]


def _stats(values: list[float] | list[int]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max())


def _iats(times: list[float]) -> list[float]:
    if len(times) < 2:
        return []
    ordered = sorted(times)
    return [ordered[i] - ordered[i - 1] for i in range(1, len(ordered))]


def extract_features(flows: list[Flow], flow_id_prefix: str = "flow") -> list[dict[str, Any]]:
    """Extract features; missing TLS fields become None (unavailable)."""
    dest_counts: dict[str, int] = {}
    src_counts: dict[str, int] = {}
    for f in flows:
        dest_counts[f.dst_ip] = dest_counts.get(f.dst_ip, 0) + 1
        src_counts[f.src_ip] = src_counts.get(f.src_ip, 0) + 1

    rows: list[dict[str, Any]] = []
    for idx, flow in enumerate(flows):
        all_lengths = flow.forward_lengths + flow.backward_lengths
        all_times = flow.forward_times + flow.backward_times
        iats = _iats(all_times)
        len_mean, len_std, len_min, len_max = _stats(all_lengths)
        iat_mean, iat_std, iat_min, iat_max = _stats(iats)

        duration = flow.duration
        total_bytes = flow.fwd_bytes + flow.bwd_bytes
        total_packets = flow.fwd_packets + flow.bwd_packets
        bps = float(total_bytes / duration) if duration > 0 else float(total_bytes)
        pps = float(total_packets / duration) if duration > 0 else float(total_packets)
        ratio = float(flow.fwd_bytes / flow.bwd_bytes) if flow.bwd_bytes > 0 else float(flow.fwd_bytes)

        tls = flow.tls_meta or {}
        cipher_suites = tls.get("tls_cipher_suites")
        alpn = tls.get("tls_alpn")

        rows.append(
            {
                "flow_id": f"{flow_id_prefix}_{idx:05d}",
                "flow_key": flow.flow_key,
                "src_ip": flow.src_ip,
                "dst_ip": flow.dst_ip,
                "src_port": flow.src_port,
                "dst_port": flow.dst_port,
                "protocol": flow.protocol,
                "duration": duration,
                "fwd_packets": flow.fwd_packets,
                "bwd_packets": flow.bwd_packets,
                "fwd_bytes": flow.fwd_bytes,
                "bwd_bytes": flow.bwd_bytes,
                "total_packets": total_packets,
                "total_bytes": total_bytes,
                "pkt_len_mean": len_mean,
                "pkt_len_std": len_std,
                "pkt_len_min": len_min,
                "pkt_len_max": len_max,
                "iat_mean": iat_mean,
                "iat_std": iat_std,
                "iat_min": iat_min,
                "iat_max": iat_max,
                "bytes_per_second": bps,
                "packets_per_second": pps,
                "fwd_bwd_byte_ratio": ratio,
                "dest_connection_count": dest_counts.get(flow.dst_ip, 1),
                "src_connection_count": src_counts.get(flow.src_ip, 1),
                "tls_present": bool(tls.get("tls_present", False)),
                "tls_version": tls.get("tls_version"),
                "tls_sni": tls.get("tls_sni"),
                "tls_alpn": ",".join(alpn) if isinstance(alpn, list) else alpn,
                "tls_cipher_suite_count": len(cipher_suites) if isinstance(cipher_suites, list) else None,
                "tls_cipher_suites_missing": bool(cipher_suites is None),
            }
        )
    return rows


def flows_to_dataframe(flows: list[Flow], flow_id_prefix: str = "flow") -> pd.DataFrame:
    rows = extract_features(flows, flow_id_prefix=flow_id_prefix)
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.DataFrame(rows)[FEATURE_COLUMNS]
