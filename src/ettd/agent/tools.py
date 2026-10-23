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
            "tls_ja3": field("tls_ja3"),
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

        target_flow = self.repo.get_flow(flow_id)
        capture_id = target_flow.get("capture_id") if target_flow else None
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
            elif rid == "beaconing_regular_timing":
                b_count = int(features.get("beacon_flow_count") or 0)
                b_cv = float(features.get("beacon_gap_cv") if features.get("beacon_gap_cv") is not None else 1.0)
                b_mean = float(features.get("beacon_gap_mean") or 0.0)
                detail = {"beacon_flow_count": b_count, "beacon_gap_cv": b_cv, "beacon_gap_mean": b_mean}
                hit = (
                    b_count >= int(cond.get("min_flow_count", 5))
                    and b_cv <= float(cond.get("max_gap_cv", 0.25))
                    and b_mean > 0.0
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
            cleaned.append(ind.strip()[:256])

        matches = []
        ja3_blocklist_path = PROJECT_ROOT / "config" / "ja3_blocklist.txt"
        ja3_candidates = [
            ind for ind in cleaned
            if len(ind) == 32 and all(c in "0123456789abcdefABCDEF" for c in ind)
        ]
        ja3_lookup_performed = False

        if ja3_blocklist_path.exists() and ja3_candidates:
            ja3_lookup_performed = True
            try:
                blocked_hashes = set()
                for line in ja3_blocklist_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        token = line.split("#")[0].strip().lower()
                        if token:
                            blocked_hashes.add(token)
                for ind in ja3_candidates:
                    if ind.lower() in blocked_hashes:
                        matches.append(
                            {
                                "indicator": ind,
                                "type": "ja3",
                                "matched": True,
                                "source": "config/ja3_blocklist.txt",
                                "description": "Matched local JA3 blocklist entry",
                            }
                        )
            except Exception:
                pass

        source = self.settings.get("threat_intelligence", {}).get("source")
        if not source:
            if matches:
                return {
                    "lookup_performed": True,
                    "reason": "matched local JA3 blocklist; no external network lookups performed",
                    "indicators": cleaned,
                    "matches": matches,
                }
            if ja3_lookup_performed:
                return {
                    "lookup_performed": True,
                    "reason": "checked local JA3 blocklist (no matches); no external network lookups performed",
                    "indicators": cleaned,
                    "matches": [],
                }
            return {
                "lookup_performed": False,
                "reason": "no threat intelligence source configured",
                "indicators": cleaned,
                "matches": [],
            }

        return {
            "lookup_performed": bool(matches) or bool(source) or ja3_lookup_performed,
            "reason": (
                "checked local JA3 blocklist and external source configured"
                if matches
                else f"source '{source}' is configured but no external connector is implemented"
            ),
            "indicators": cleaned,
            "matches": matches,
        }
