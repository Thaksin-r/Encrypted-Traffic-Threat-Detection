"""Numeric feature matrix helpers for ML (leak-aware column selection)."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURE_COLUMNS = [
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
    "tls_cipher_suite_count",
]


def build_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in df.columns]
    frame = df[cols].copy()
    if "tls_present" in frame.columns:
        frame["tls_present"] = frame["tls_present"].astype(float)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame.to_numpy(dtype=float), cols


def group_ids(df: pd.DataFrame) -> pd.Series:
    """Group related flows to reduce train/test contamination."""
    if "flow_key" in df.columns:
        return df["flow_key"].astype(str)
    if "dst_ip" in df.columns and "src_ip" in df.columns:
        return df["src_ip"].astype(str) + "->" + df["dst_ip"].astype(str)
    return df["flow_id"].astype(str)
