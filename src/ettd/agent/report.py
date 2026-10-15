"""Structured investigation report schema helpers."""

from __future__ import annotations

from typing import Any


REQUIRED_REPORT_FIELDS = [
    "incident_summary",
    "observed_evidence",
    "tools_called",
    "ml_prediction",
    "likely_explanation",
    "alternative_explanations",
    "confidence",
    "missing_evidence_and_limitations",
    "recommended_next_steps",
]


def empty_report(flow_id: str) -> dict[str, Any]:
    return {
        "flow_id": flow_id,
        "incident_summary": "",
        "observed_evidence": [],
        "tools_called": [],
        "ml_prediction": {},
        "likely_explanation": "",
        "alternative_explanations": [],
        "confidence": {
            "value": "low",
            "definition": (
                "Qualitative confidence based on quantity/quality of *observed* evidence "
                "and tool agreement. Not a calibrated probability."
            ),
        },
        "missing_evidence_and_limitations": [],
        "recommended_next_steps": [],
        "facts_vs_hypotheses_note": (
            "Observed evidence lists tool-backed facts. Explanations are hypotheses."
        ),
    }


def validate_report(report: dict[str, Any]) -> list[str]:
    missing = [f for f in REQUIRED_REPORT_FIELDS if f not in report]
    return missing


def build_report_from_tool_results(
    flow_id: str,
    tool_calls: list[dict[str, Any]],
    prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deterministic evidence-backed report used when LLM is unavailable for tests,
    and as a scaffold the LLM report must still respect."""
    report = empty_report(flow_id)
    evidence = []
    matched_rules = []
    related_count = 0
    tls_info = {}

    for call in tool_calls:
        name = call.get("tool")
        result = call.get("result") or {}
        report["tools_called"].append(
            {
                "tool": name,
                "arguments": call.get("arguments"),
                "findings": result,
            }
        )
        if result.get("error"):
            evidence.append({"type": "error", "tool": name, "detail": result["error"]})
            continue
        if name == "inspect_flow" and "features" in result:
            feats = result["features"]
            evidence.append(
                {
                    "type": "flow_metadata",
                    "src_ip": feats.get("src_ip"),
                    "dst_ip": feats.get("dst_ip"),
                    "dst_port": feats.get("dst_port"),
                    "protocol": feats.get("protocol"),
                    "duration": feats.get("duration"),
                    "total_bytes": feats.get("total_bytes"),
                    "total_packets": feats.get("total_packets"),
                }
            )
        if name == "analyze_tls_metadata":
            tls_info = result
            evidence.append({"type": "tls_metadata", "detail": result})
        if name == "find_related_flows":
            related_count = int(result.get("related_count") or 0)
            evidence.append({"type": "related_flows", "count": related_count})
        if name == "compare_detection_rules":
            matched_rules = result.get("matched_rules") or []
            evidence.append({"type": "rule_matches", "matched": matched_rules})
        if name == "check_threat_intelligence":
            evidence.append(
                {
                    "type": "threat_intelligence",
                    "lookup_performed": result.get("lookup_performed"),
                    "reason": result.get("reason"),
                }
            )

    report["observed_evidence"] = evidence
    report["ml_prediction"] = prediction or {}
    score = (prediction or {}).get("score")
    label = (prediction or {}).get("label", "unknown")

    if matched_rules:
        report["incident_summary"] = (
            f"Flow {flow_id} labeled {label} by ML; {len(matched_rules)} behavioral rule(s) matched."
        )
        report["likely_explanation"] = (
            "Suspicious behavioral patterns in encrypted-traffic metadata "
            f"(rules: {[m.get('rule_id') for m in matched_rules]})."
        )
        report["confidence"]["value"] = "medium"
    else:
        report["incident_summary"] = (
            f"Flow {flow_id} labeled {label} by ML; no configured behavioral rules matched."
        )
        report["likely_explanation"] = (
            "ML flagged the flow based on feature patterns; no independent rule corroboration."
        )
        report["confidence"]["value"] = "low"

    report["alternative_explanations"] = [
        "Benign bulk transfer or software update with asymmetric byte ratios.",
        "Scanning or misconfiguration producing short bursts.",
        "Incomplete PCAP visibility causing missing TLS handshake metadata.",
    ]
    report["missing_evidence_and_limitations"] = [
        "Application payloads were not decrypted and are unavailable.",
        "ML score is not a calibrated probability.",
        "No endpoint telemetry or authenticated user context.",
        f"Related flows observed in capture: {related_count}.",
    ]
    if tls_info:
        report["missing_evidence_and_limitations"].append(
            f"TLS SNI field: {tls_info.get('tls_sni')}; version: {tls_info.get('tls_version')}."
        )
    report["recommended_next_steps"] = [
        "Validate authorization and source of the PCAP before further action.",
        "Correlate destination IP/domain with approved asset inventory.",
        "If policy allows, obtain endpoint logs for the same time window.",
        "Do not auto-block; escalate for human review if rules and ML agree.",
    ]
    if score is not None:
        report["ml_prediction"] = {
            **(prediction or {}),
            "score_interpretation": (prediction or {}).get(
                "score_note",
                "Score is a ranking signal, not a calibrated probability.",
            ),
        }
    return report
