"""Streamlit dashboard for Encrypted Traffic Threat Detection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("ETTD_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="ETTD — Encrypted Traffic Threat Detection",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0b1220 0%, #121a2b 40%, #0e1624 100%); color: #e8eef7; }
    h1, h2, h3 { color: #f4f7fb !important; letter-spacing: 0.02em; }
    div[data-testid="stMetricValue"] { color: #7dd3fc; }
    .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, params: dict | None = None):
    with httpx.Client(base_url=st.session_state.api_url, timeout=60.0) as client:
        r = client.get(path, params=params)
        r.raise_for_status()
        return r.json()


def api_post(path: str, payload: dict | None = None):
    with httpx.Client(base_url=st.session_state.api_url, timeout=120.0) as client:
        r = client.post(path, json=payload or {})
        r.raise_for_status()
        return r.json()


if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API

st.title("ETTD")
st.caption("Encrypted traffic threat detection from metadata — no decryption, no auto-blocking.")

with st.sidebar:
    st.header("Connection")
    st.session_state.api_url = st.text_input("API URL", st.session_state.api_url)
    if st.button("Refresh"):
        st.rerun()
    st.markdown("---")
    st.markdown("**Security notes**")
    st.markdown("- Offline authorized PCAPs only")
    st.markdown("- Payloads are never decrypted")
    st.markdown("- Agent tools are read-only")

try:
    summary = api_get("/summary")
except Exception as exc:
    st.error(f"Cannot reach API at {st.session_state.api_url}: {exc}")
    st.info("Start the API with: `uvicorn ettd.api.main:app --reload --app-dir src`")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flows analyzed", summary.get("flows_analyzed", 0))
c2.metric("Suspicious", summary.get("suspicious", 0))
c3.metric("Benign", summary.get("benign", 0))
c4.metric("Investigations", summary.get("investigations", 0))

tab_flows, tab_actions, tab_inv = st.tabs(["Flows", "Actions", "Investigations"])

with tab_actions:
    st.subheader("Pipeline actions")
    pcap_path = st.text_input("Local PCAP path (server-side)", "data/samples/sample.pcap")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Extract features"):
            with httpx.Client(base_url=st.session_state.api_url, timeout=120.0) as client:
                r = client.post("/captures/extract", params={"pcap_path": pcap_path})
                r.raise_for_status()
                result = r.json()
            st.success(result)
    with col_b:
        if st.button("Run ML detection"):
            result = api_post("/detect", {"retrain": True})
            st.success(
                f"Mode={result.get('mode')} suspicious={result.get('suspicious')} benign={result.get('benign')}"
            )
    with col_c:
        flow_id = st.text_input("Flow ID to investigate")
        use_llm = st.checkbox("Use LLM agent", value=False, help="Requires OPENAI_API_KEY on the API server")
        if st.button("Investigate") and flow_id:
            result = api_post("/investigate", {"flow_id": flow_id, "use_llm": use_llm})
            st.json(result)

with tab_flows:
    st.subheader("Flow details")
    try:
        flows = api_get("/flows")
    except Exception as exc:
        st.error(exc)
        flows = []
    if not flows:
        st.info("No flows yet. Extract a PCAP from the Actions tab.")
    else:
        for flow in flows:
            pred = flow.get("prediction") or {}
            label = pred.get("label", "n/a")
            score = pred.get("score")
            title = f"{flow['flow_id']} | {flow['src_ip']} → {flow['dst_ip']}:{flow['dst_port']} | {label}"
            with st.expander(title):
                st.write("**Metadata / features**")
                st.json(flow.get("features") or {})
                st.write("**ML prediction**")
                st.json(pred if pred else {"status": "not scored"})
                if score is not None:
                    st.caption("Score is a ranking signal, not a calibrated probability.")

with tab_inv:
    st.subheader("Investigation reports")
    investigations = api_get("/investigations")
    if not investigations:
        st.info("No investigations yet.")
    for inv in investigations:
        with st.expander(f"{inv['investigation_id']} | {inv['flow_id']} | {inv['status']}"):
            st.write("**Status:**", inv["status"])
            st.write("**Tool calls**")
            st.json(inv.get("tool_calls") or inv.get("tool_calls_json") or [])
            st.write("**Report**")
            report = inv.get("report")
            if report:
                st.json(report)
            else:
                st.warning("No report available.")
