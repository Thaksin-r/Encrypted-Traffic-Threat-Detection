"""Read-only investigation tools with validated arguments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ettd.config import PROJECT_ROOT, load_settings, resolve_path
from ettd.db.repository import Repository

_FLOW_ID_RE = re.compile(r"^[A-Za-z0-9_\-:.]{1,128}$")


def _validate_flow_id(flow_id: str) -> str:
    if not isinstance(flow_id, str) or not _FLOW_ID_RE.match(flow_id):
        raise ValueError("invalid flow_id: must be 1-128 safe alphanumeric characters")
    return flow_id


class InvestigationTools:
    """Approved analysis functions. No shell, no network mutation."""

    def __init__(self, repo: Repository, settings: dict[str, Any] | None = None) -> None:
        self.repo = repo
        self.settings = settings or load_settings()

    def inspect_flow(self, flow_id: str) -> dict[str, Any]:
        flow_id = _validate_flow_id(flow_id)
        features = self.repo.get_flow_features(flow_id)
        if features is None:
            return {"error": "flow_not_found", "flow_id": flow_id}
        prediction = self.repo.get_prediction(flow_id)
        return {
            "flow_id": flow_id,
            "features": features,
            "prediction": prediction,
            "note": "Observed metadata and model output only; payloads were not decrypted.",
        }

    def analyze_tls_metadata(self, flow_id: str) -> dict[str, Any]:
        flow_id = _validate_flow_id(flow_id)
        features = self.repo.get_flow_features(flow_id)
        if features is None:
            return {"error": "flow_not_found", "flow_id": flow_id}

        def field(name: str):
            value = features.get(name)
            if value is None or value == "" or (isinstance(value, float) and value != value):
                return "unavailable"
            return value

        return {
            "flow_id": flow_id,
            "tls_present": bool(features.get("tls_present")),
            "tls_version": field("tls_version"),
            "tls_sni": field("tls_sni"),
            "tls_alpn": field("tls_alpn"),
            "tls_cipher_suite_count": field("tls_cipher_suite_count"),
            "limitations": (
                "Only cleartext handshake observables are considered. "
                "Encrypted application data is never decrypted."
            ),
        }

    def find_related_flows(self, flow_id: str) -> dict[str, Any]:
        flow_id = _validate_flow_id(flow_id)
        features = self.repo.get_flow_features(flow_id)
        if features is None:
            return {"error": "flow_not_found", "flow_id": flow_id}

        capture_id = None
        for flow in self.repo.list_flows():
            if flow["flow_id"] == flow_id:
                capture_id = flow["capture_id"]
                break
        candidates = self.repo.list_flows(capture_id)
        related = []
        for flow in candidates:
            if flow["flow_id"] == flow_id:
                continue
            score = 0
            reasons = []
            if flow["dst_ip"] == features.get("dst_ip"):
                score += 2
                reasons.append("same_dst_ip")
            if flow["src_ip"] == features.get("src_ip"):
                score += 1
                reasons.append("same_src_ip")
            if flow["dst_port"] == features.get("dst_port"):
                score += 1
                reasons.append("same_dst_port")
            if score:
                related.append(
                    {
                        "flow_id": flow["flow_id"],
                        "relation_score": score,
                        "reasons": reasons,
                        "dst_ip": flow["dst_ip"],
                        "dst_port": flow["dst_port"],
                        "protocol": flow["protocol"],
                    }
                )
        related.sort(key=lambda r: r["relation_score"], reverse=True)
        return {
            "flow_id": flow_id,
            "related_count": len(related),
            "related_flows": related[:25],
            "note": "Relatedness uses observable metadata only.",
        }

    def compare_detection_rules(self, flow_id: str) -> dict[str, Any]:
        flow_id = _validate_flow_id(flow_id)
        features = self.repo.get_flow_features(flow_id)
        if features is None:
            return {"error": "flow_not_found", "flow_id": flow_id}

        rules_path = resolve_path(self.settings.get("detection_rules_path")) or (
            PROJECT_ROOT / "config" / "detection_rules.yaml"
        )
        with Path(rules_path).open("r", encoding="utf-8") as fh:
            rules_doc = yaml.safe_load(fh) or {}
        rules = rules_doc.get("rules", [])
        matched = []
        checked = []

        for rule in rules:
            if not rule.get("enabled", True):
                continue
            rid = rule.get("id", "unknown")
            checked.append(rid)
            cond = rule.get("conditions", {})
            hit = False
            detail = {}

            if rid == "high_byte_ratio":
                total = float(features.get("total_bytes") or 0)
                fwd = float(features.get("fwd_bytes") or 0)
                bwd = float(features.get("bwd_bytes") or 0)
                minor = min(fwd, bwd)
                ratio = minor / total if total else 1.0
                detail = {"total_bytes": total, "minor_direction_ratio": ratio}
                hit = total >= float(cond.get("min_total_bytes", 5000)) and ratio <= float(
                    cond.get("max_minor_direction_ratio", 0.05)
                )
            elif rid == "short_burst_many_packets":
                duration = float(features.get("duration") or 0)
                pkts = int(features.get("total_packets") or 0)
                detail = {"duration": duration, "total_packets": pkts}
                hit = duration <= float(cond.get("max_duration_seconds", 0.5)) and pkts >= int(
                    cond.get("min_total_packets", 40)
                )
            elif rid == "repeated_same_dest":
                count = int(features.get("dest_connection_count") or 0)
                detail = {"dest_connection_count": count}
                hit = count >= int(cond.get("min_dest_connection_count", 8))
            elif rid == "missing_tls_on_https_port":
                dport = int(features.get("dst_port") or 0)
                tls = bool(features.get("tls_present"))
                detail = {"dst_port": dport, "tls_present": tls}
                hit = dport == int(cond.get("dest_port", 443)) and (not tls) and bool(
                    cond.get("require_tls_metadata", True)
                )

            if hit:
                matched.append(
                    {
                        "rule_id": rid,
                        "description": rule.get("description"),
                        "detail": detail,
                    }
                )

        return {
            "flow_id": flow_id,
            "rules_checked": checked,
            "matched_rules": matched,
            "note": "Rules are transparent heuristics, not proof of malice.",
        }

    def check_threat_intelligence(self, indicators: list[str] | str) -> dict[str, Any]:
        if isinstance(indicators, str):
            indicators = [indicators]
        if not isinstance(indicators, list):
            raise ValueError("indicators must be a list of strings")
        cleaned = []
        for ind in indicators[:50]:
            if not isinstance(ind, str) or len(ind) > 256:
                raise ValueError("each indicator must be a string <= 256 chars")
            # Strip control chars; treat as untrusted
            cleaned.append(ind.strip()[:256])

        source = self.settings.get("threat_intelligence", {}).get("source")
        if not source:
            return {
                "lookup_performed": False,
                "reason": "no threat intelligence source configured",
                "indicators": cleaned,
                "matches": [],
            }
        # Intentionally no automatic external lookups in MVP without explicit config wiring.
        return {
            "lookup_performed": False,
            "reason": f"source '{source}' is configured but no connector is implemented in MVP",
            "indicators": cleaned,
            "matches": [],
        }
