"""Download and ingest the UNSW-NB15 benchmark network dataset into ETTD SQLite DB."""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from ettd.config import PROJECT_ROOT, db_path, load_settings
from ettd.db.repository import Repository

logger = logging.getLogger(__name__)

UNSW_URL = "https://raw.githubusercontent.com/ushukkla/nospammers/master/UNSW_NB15_training-set.csv"


def download_and_ingest_unsw(
    num_samples: int = 1500,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Download UNSW-NB15 flow records and map them into ETTD database."""
    settings = settings or load_settings()
    repo = Repository(db_path(settings))

    logger.info("Streaming UNSW-NB15 dataset from %s (target=%d flows)...", UNSW_URL, num_samples)
    
    with httpx.Client(timeout=45.0) as client:
        with client.stream("GET", UNSW_URL) as response:
            response.raise_for_status()
            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line)
                if len(lines) >= num_samples + 1:  # header + samples
                    break

    csv_text = "\n".join(lines)
    raw_df = pd.read_csv(io.StringIO(csv_text))
    
    batch_tag = datetime.now(timezone.utc).strftime("%m%d_%H%M%S")
    capture_id = f"unsw_nb15_{batch_tag}"
    repo.create_capture(source_path="remote:UNSW_NB15_training-set.csv", capture_id=capture_id)

    # Synthetic realistic IP pool to map IDs
    flows_records: list[dict[str, Any]] = []
    labels_records: list[dict[str, Any]] = []

    # Common port mapping based on service
    service_port_map = {
        "http": 80,
        "ssl": 443,
        "dns": 53,
        "ftp": 21,
        "ftp-data": 20,
        "smtp": 25,
        "ssh": 22,
        "dhcp": 67,
        "snmp": 161,
        "pop3": 110,
    }

    base_time = 1700000000.0

    for idx, row in raw_df.iterrows():
        flow_id = f"unsw_{batch_tag}_{idx:05d}"
        
        # IP assignments based on internal subnet and internet targets
        proto = str(row.get("proto", "tcp")).upper()
        if proto not in ("TCP", "UDP", "ICMP"):
            proto = "TCP"

        srv = str(row.get("service", "-")).lower()
        dst_port = service_port_map.get(srv)
        if dst_port is None:
            dst_port = 443 if (idx % 3 == 0) else (80 if idx % 2 == 0 else 53)

        src_host_id = (idx % 15) + 10
        dst_host_id = (idx % 25) + 1
        src_ip = f"192.168.1.{src_host_id}"
        dst_ip = f"203.0.113.{dst_host_id}" if idx % 2 == 0 else f"198.51.100.{dst_host_id}"
        src_port = 40000 + (idx % 20000)

        dur = max(0.001, float(row.get("dur", 0.05)))
        start_t = base_time + (idx * 0.5)
        end_t = start_t + dur

        spkts = int(row.get("spkts", 1))
        dpkts = int(row.get("dpkts", 0))
        sbytes = int(row.get("sbytes", 100))
        dbytes = int(row.get("dbytes", 0))
        total_packets = spkts + dpkts
        total_bytes = sbytes + dbytes

        rate = float(row.get("rate", total_packets / dur))
        bps = float((sbytes + dbytes) / dur) if dur > 0 else float(total_bytes)
        pps = float(total_packets / dur) if dur > 0 else float(total_packets)

        sinpkt = float(row.get("sinpkt", 0.001))
        dinpkt = float(row.get("dinpkt", 0.001))
        iat_mean = (sinpkt + dinpkt) / 2.0
        iat_std = abs(sinpkt - dinpkt) / 2.0

        smean = float(row.get("smean", sbytes / max(1, spkts)))
        dmean = float(row.get("dmean", dbytes / max(1, dpkts)))
        pkt_len_mean = (sbytes + dbytes) / max(1, total_packets)
        pkt_len_min = min(smean, dmean)
        pkt_len_max = max(smean, dmean) * 1.5
        pkt_len_std = abs(pkt_len_max - pkt_len_min) / 4.0

        fwd_bwd_byte_ratio = float(sbytes / max(1, dbytes))

        # Beaconing indicators from inter-packet times
        beacon_gap_mean = float(sinpkt)
        beacon_gap_cv = float(iat_std / (iat_mean + 1e-9))

        # TLS flags
        is_tls = bool(dst_port == 443 or srv == "ssl" or idx % 4 == 0)
        sni = f"api-endpoint-{dst_host_id}.cloud-service.net" if is_tls else None
        ja3 = "771,4865-4866-4867,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-21,29-23-24,0" if is_tls else None

        flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"

        is_attack = int(row.get("label", 0)) == 1
        attack_category = str(row.get("attack_cat", "Normal")).strip()

        feature_record = {
            "flow_id": flow_id,
            "flow_key": flow_key,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": proto,
            "duration": dur,
            "fwd_packets": spkts,
            "bwd_packets": dpkts,
            "fwd_bytes": sbytes,
            "bwd_bytes": dbytes,
            "total_packets": total_packets,
            "total_bytes": total_bytes,
            "pkt_len_mean": pkt_len_mean,
            "pkt_len_std": pkt_len_std,
            "pkt_len_min": pkt_len_min,
            "pkt_len_max": pkt_len_max,
            "iat_mean": iat_mean,
            "iat_std": iat_std,
            "iat_min": min(sinpkt, dinpkt),
            "iat_max": max(sinpkt, dinpkt),
            "bytes_per_second": bps,
            "packets_per_second": pps,
            "fwd_bwd_byte_ratio": fwd_bwd_byte_ratio,
            "dest_connection_count": int(row.get("ct_dst_ltm", 1)),
            "src_connection_count": int(row.get("ct_src_dport_ltm", 1)),
            "beacon_gap_mean": beacon_gap_mean,
            "beacon_gap_cv": beacon_gap_cv,
            "beacon_flow_count": int(row.get("ct_srv_src", 1)),
            "tls_present": is_tls,
            "tls_version": "TLS 1.3" if is_tls else None,
            "tls_sni": sni,
            "tls_alpn": "h2,http/1.1" if is_tls else None,
            "tls_cipher_suite_count": 15 if is_tls else None,
            "tls_cipher_suites_missing": not is_tls,
            "tls_ja3": ja3,
            "attack_category": attack_category,
        }
        flows_records.append(feature_record)
        labels_records.append({
            "flow_id": flow_id,
            "label": "suspicious" if is_attack else "benign",
            "attack_cat": attack_category,
        })

    features_df = pd.DataFrame(flows_records)
    flow_times = {
        r["flow_id"]: (base_time + (i * 0.5), base_time + (i * 0.5) + r["duration"])
        for i, r in enumerate(flows_records)
    }

    # Save to database
    repo.save_flows_and_features(capture_id, features_df, flow_times=flow_times)

    # Save labels file
    labels_dir = PROJECT_ROOT / "data" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_file = labels_dir / "unsw_nb15_labels.csv"
    labels_df = pd.DataFrame(labels_records)
    labels_df[["flow_id", "label"]].to_csv(labels_file, index=False)

    attacks_summary = labels_df["attack_cat"].value_counts().to_dict()

    return {
        "status": "success",
        "capture_id": capture_id,
        "total_flows": len(features_df),
        "attacks": int(labels_df[labels_df["label"] == "suspicious"].shape[0]),
        "benign": int(labels_df[labels_df["label"] == "benign"].shape[0]),
        "attack_breakdown": attacks_summary,
        "labels_path": str(labels_file),
    }
