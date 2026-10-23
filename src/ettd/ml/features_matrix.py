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
    "beacon_gap_mean",
    "beacon_gap_cv",
    "beacon_flow_count",
    "tls_present",
    "tls_cipher_suite_count",
    "tls_cipher_suites_missing",
]


def build_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    frame = df.copy()
    if "tls_cipher_suites_missing" not in frame.columns:
        if "tls_cipher_suite_count" in frame.columns:
            frame["tls_cipher_suites_missing"] = frame["tls_cipher_suite_count"].isna().astype(float)
        else:
            frame["tls_cipher_suites_missing"] = 1.0

    cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in frame.columns]
    mat = frame[cols].copy()
    if "tls_present" in mat.columns:
        mat["tls_present"] = mat["tls_present"].astype(float)
    if "tls_cipher_suites_missing" in mat.columns:
        mat["tls_cipher_suites_missing"] = mat["tls_cipher_suites_missing"].astype(float)
    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return mat.to_numpy(dtype=float), cols


def group_ids(df: pd.DataFrame) -> pd.Series:
    """Group related flows to reduce train/test contamination."""
    if "flow_key" in df.columns:
        return df["flow_key"].astype(str)
    if "dst_ip" in df.columns and "src_ip" in df.columns:
        return df["src_ip"].astype(str) + "->" + df["dst_ip"].astype(str)
    return df["flow_id"].astype(str)
