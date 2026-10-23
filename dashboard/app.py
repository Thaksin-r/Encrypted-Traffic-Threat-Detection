"""Streamlit dashboard for Encrypted Traffic Threat Detection (ETTD)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import httpx
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DEFAULT_API = os.environ.get("ETTD_API_URL", "http://127.0.0.1:8000")

# Color palette: Suspicious is bright red/rose, Benign is emerald green, Accent is cyan
COLOR_SUSPICIOUS = "#ef4444"
COLOR_BENIGN = "#10b981"
COLOR_ACCENT = "#38bdf8"
COLOR_MUTED = "#64748b"
COLOR_BG = "#0b1220"

st.set_page_config(
    page_title="ETTD — Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark glassmorphism styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #121a2b 40%, #0e1624 100%);
        color: #e8eef7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    h1, h2, h3, h4 { color: #f4f7fb !important; letter-spacing: 0.02em; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #7dd3fc; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9rem; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
    .glass-card {
        background: rgba(18, 26, 43, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .badge-suspicious {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-benign {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid #10b981;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 500;
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API


# Cached API Helpers
@st.cache_data(ttl=5)
def api_get(api_url: str, path: str, params: tuple | None = None) -> Any:
    query_dict = dict(params) if params else None
    with httpx.Client(base_url=api_url, timeout=30.0) as client:
        r = client.get(path, params=query_dict)
        r.raise_for_status()
        return r.json()


def api_post(api_url: str, path: str, payload: dict | None = None, params: dict | None = None) -> Any:
    with httpx.Client(base_url=api_url, timeout=120.0) as client:
        r = client.post(path, json=payload or {}, params=params)
        r.raise_for_status()
        return r.json()


def api_upload(api_url: str, path: str, files: dict) -> Any:
    with httpx.Client(base_url=api_url, timeout=120.0) as client:
        r = client.post(path, files=files)
        r.raise_for_status()
        return r.json()


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#cbd5e1", family="sans-serif"),
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    )
    return fig


# Sidebar
with st.sidebar:
    st.title("🛡️ ETTD")
    st.caption("Offline Metadata Threat Detection")
    st.session_state.api_url = st.text_input("Backend API URL", st.session_state.api_url)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Operational Guardrails")
    st.markdown("- **Zero Decryption:** Payload contents are strictly ignored.")
    st.markdown("- **Read-Only Agent:** Investigation tools cannot alter states or firewalls.")
    st.markdown("- **No Auto-Blocking:** Decisions remain human-in-the-loop.")


# Verify Connection & Fetch Overview KPI Summary
try:
    summary = api_get(st.session_state.api_url, "/summary")
except Exception as exc:
    st.error(f"Cannot connect to ETTD Backend at `{st.session_state.api_url}`")
    st.info("Start the API using: `uvicorn ettd.api.main:app --app-dir src --reload`")
    st.stop()

# Header & KPIs
st.title("Encrypted Traffic Threat Detection")
st.caption("Statistical anomaly detection & rule-based inspection across encrypted flows without decryption.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Flows Analyzed", summary.get("flows_analyzed", 0))
kpi2.metric("Suspicious Flows", summary.get("suspicious", 0), delta=f"{summary.get('suspicious', 0)} flagged", delta_color="inverse")
kpi3.metric("Benign Flows", summary.get("benign", 0))
kpi4.metric("Investigations", summary.get("investigations", 0))

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_overview, tab_flows, tab_network, tab_investigations, tab_actions = st.tabs(
    ["📊 Overview", "🌊 Flows", "🕸️ Network", "🔍 Investigations", "⚡ Actions"]
)

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tab_overview:
    col_hist, col_time = st.columns(2)

    with col_hist:
        st.subheader("Score Distribution")
        try:
            score_dist = api_get(st.session_state.api_url, "/stats/score-distribution")
            scores = score_dist.get("scores", [])
            thresh = score_dist.get("threshold", 0.55)

            if scores:
                fig_hist = px.histogram(
                    x=scores,
                    nbins=30,
                    title=f"Flow Anomaly Scores (Threshold = {thresh:.3f})",
                    labels={"x": "Score (Higher = More Anomalous)"},
                    color_discrete_sequence=[COLOR_ACCENT],
                )
                fig_hist.add_vline(
                    x=thresh,
                    line_dash="dash",
                    line_color=COLOR_SUSPICIOUS,
                    annotation_text="Threshold",
                    annotation_position="top right",
                )
                apply_dark_theme(fig_hist)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No score distribution data available yet.")
        except Exception as e:
            st.warning(f"Could not load score distribution: {e}")

    with col_time:
        st.subheader("Flow Timeline")
        try:
            timeline_data = api_get(st.session_state.api_url, "/stats/timeline", (("bucket_seconds", 60.0),))
            if timeline_data:
                df_time = pd.DataFrame(timeline_data)
                df_time["time_dt"] = pd.to_datetime(df_time["timestamp"], unit="s")
                df_time["benign_flows"] = df_time["total_flows"] - df_time["suspicious_flows"]

                fig_time = go.Figure()
                fig_time.add_trace(
                    go.Bar(
                        x=df_time["time_dt"],
                        y=df_time["benign_flows"],
                        name="Benign",
                        marker_color=COLOR_BENIGN,
                    )
                )
                fig_time.add_trace(
                    go.Bar(
                        x=df_time["time_dt"],
                        y=df_time["suspicious_flows"],
                        name="Suspicious",
                        marker_color=COLOR_SUSPICIOUS,
                    )
                )
                fig_time.update_layout(
                    barmode="stack",
                    title="Traffic Volume & Threat Detections Over Time",
                    xaxis_title="Time",
                    yaxis_title="Flows Count",
                )
                apply_dark_theme(fig_time)
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("No timeline events captured yet.")
        except Exception as e:
            st.warning(f"Could not load timeline: {e}")

    col_talkers, col_proto = st.columns(2)

    with col_talkers:
        st.subheader("Top Destinations")
        try:
            top_talkers = api_get(st.session_state.api_url, "/stats/top-talkers", (("limit", 10),))
            by_flows = top_talkers.get("by_flows", [])
            by_bytes = top_talkers.get("by_bytes", [])

            metric_choice = st.radio("Metric", ["Flow Count", "Total Bytes"], horizontal=True, label_visibility="collapsed")
            if metric_choice == "Flow Count" and by_flows:
                df_tt = pd.DataFrame(by_flows)
                fig_tt = px.bar(
                    df_tt,
                    x="flow_count",
                    y="dst_ip",
                    orientation="h",
                    title="Top Destinations by Flow Count",
                    labels={"flow_count": "Flows", "dst_ip": "Destination IP"},
                    color_discrete_sequence=["#38bdf8"],
                )
                fig_tt.update_layout(yaxis=dict(autorange="reversed"))
                apply_dark_theme(fig_tt)
                st.plotly_chart(fig_tt, use_container_width=True)
            elif metric_choice == "Total Bytes" and by_bytes:
                df_tt = pd.DataFrame(by_bytes)
                fig_tt = px.bar(
                    df_tt,
                    x="total_bytes",
                    y="dst_ip",
                    orientation="h",
                    title="Top Destinations by Total Bytes",
                    labels={"total_bytes": "Bytes Transferred", "dst_ip": "Destination IP"},
                    color_discrete_sequence=["#a855f7"],
                )
                fig_tt.update_layout(yaxis=dict(autorange="reversed"))
                apply_dark_theme(fig_tt)
                st.plotly_chart(fig_tt, use_container_width=True)
            else:
                st.info("No destination metrics available.")
        except Exception as e:
            st.warning(f"Could not load top destinations: {e}")

    with col_proto:
        st.subheader("Protocol & Port Distribution")
        try:
            flows_sample = api_get(st.session_state.api_url, "/flows", (("limit", 500),))
            if flows_sample:
                df_sample = pd.DataFrame(flows_sample)
                port_counts = df_sample["dst_port"].value_counts().reset_index()
                port_counts.columns = ["port", "count"]
                port_counts["label"] = port_counts["port"].apply(
                    lambda p: f"Port {p} (HTTPS)" if p in (443, 8443) else f"Port {p}"
                )

                fig_donut = px.pie(
                    port_counts.head(8),
                    names="label",
                    values="count",
                    hole=0.55,
                    title="Destination Port Share",
                    color_discrete_sequence=px.colors.sequential.Tealgrn,
                )
                apply_dark_theme(fig_donut)
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("No flow samples available for protocol distribution.")
        except Exception as e:
            st.warning(f"Could not load port share: {e}")


# ==============================================================================
# TAB 2: FLOWS
# ==============================================================================
with tab_flows:
    st.subheader("Flow Directory & Deep Inspection")

    # Filters row
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([1.5, 1.5, 1.5, 2.5, 1.2])
    with fcol1:
        f_label = st.selectbox("Label Filter", ["All", "suspicious", "benign"])
    with fcol2:
        f_min_score = st.slider("Min Anomaly Score", 0.0, 1.0, 0.0, step=0.05)
    with fcol3:
        f_port_str = st.text_input("Destination Port", "")
        f_port = int(f_port_str) if f_port_str.strip().isdigit() else None
    with fcol4:
        f_search = st.text_input("Search IP / SNI", "")
    with fcol5:
        f_limit = st.selectbox("Page Size", [25, 50, 100], index=0)

    # Prepare query params
    params = [
        ("limit", f_limit),
        ("offset", 0),
        ("sort_by", "score"),
        ("sort_order", "desc"),
    ]
    if f_label != "All":
        params.append(("label", f_label))
    if f_min_score > 0:
        params.append(("min_score", f_min_score))
    if f_port:
        params.append(("dst_port", f_port))
    if f_search.strip():
        params.append(("search", f_search.strip()))

    try:
        flow_records = api_get(st.session_state.api_url, "/flows", tuple(params))
    except Exception as exc:
        st.error(f"Error fetching flows: {exc}")
        flow_records = []

    if not flow_records:
        st.info("No matching flows found. Try adjusting filters or extracting a PCAP in the Actions tab.")
    else:
        # Build clean display table
        table_rows = []
        for fl in flow_records:
            pred = fl.get("prediction") or {}
            feat = fl.get("features") or {}
            table_rows.append(
                {
                    "flow_id": fl.get("flow_id"),
                    "src": f"{fl.get('src_ip')}:{fl.get('src_port')}",
                    "dst:port": f"{fl.get('dst_ip')}:{fl.get('dst_port')}",
                    "protocol": "TCP" if fl.get("protocol") == 6 else ("UDP" if fl.get("protocol") == 17 else str(fl.get("protocol"))),
                    "bytes": int(fl.get("bytes") or 0),
                    "packets": int(fl.get("packets") or 0),
                    "duration": round(float(fl.get("duration") or 0.0), 3),
                    "TLS SNI": feat.get("tls_sni") or "-",
                    "label": pred.get("label", "unscored"),
                    "score": round(float(pred.get("score") or 0.0), 4) if pred.get("score") is not None else None,
                }
            )

        df_display = pd.DataFrame(table_rows)

        # Style labels with colors
        def highlight_label(val):
            if val == "suspicious":
                return f"color: {COLOR_SUSPICIOUS}; font-weight: bold;"
            elif val == "benign":
                return f"color: {COLOR_BENIGN}; font-weight: bold;"
            return "color: #94a3b8;"

        styled_df = df_display.style.map(highlight_label, subset=["label"])
        st.dataframe(styled_df, use_container_width=True, height=280)

        # Selection for deep inspection panel
        st.markdown("---")
        st.markdown("### Selected Flow Inspection")
        flow_id_list = [r["flow_id"] for r in flow_records]
        selected_flow_id = st.selectbox("Select Flow ID to Inspect", flow_id_list)

        if selected_flow_id:
            try:
                flow_detail = api_get(st.session_state.api_url, f"/flows/{selected_flow_id}")
            except Exception as e:
                st.error(f"Could not load flow detail: {e}")
                flow_detail = None

            if flow_detail:
                feat = flow_detail.get("features") or {}
                pred = flow_detail.get("prediction") or {}

                detail_col1, detail_col2 = st.columns([1.5, 1])

                with detail_col1:
                    st.markdown("#### Feature Deviation & Explanation")
                    try:
                        explain_data = api_get(st.session_state.api_url, f"/flows/{selected_flow_id}/explain")
                        top_features = explain_data.get("top_features", [])
                        if top_features:
                            df_exp = pd.DataFrame(top_features)
                            fig_exp = px.bar(
                                df_exp,
                                x="score",
                                y="feature",
                                orientation="h",
                                title=f"Top Deviant Features ({explain_data.get('mode')})",
                                labels={"score": "Deviation / Importance Score", "feature": "Feature"},
                                color="score",
                                color_continuous_scale="Reds",
                            )
                            fig_exp.update_layout(yaxis=dict(autorange="reversed"))
                            apply_dark_theme(fig_exp)
                            st.plotly_chart(fig_exp, use_container_width=True)
                        else:
                            st.info("No feature explanation available for this flow.")
                    except Exception as e:
                        st.info(f"Feature explanation unavailable: {e}")

                with detail_col2:
                    st.markdown("#### Metadata & Threat Intel")
                    st.markdown(f"**Flow ID:** `{selected_flow_id}`")
                    st.markdown(
                        f"**Classification:** <span class='badge-{pred.get('label', 'benign')}'>{pred.get('label', 'unscored').upper()}</span> "
                        f"(Score: `{pred.get('score', 'N/A')}`)",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**TLS SNI:** `{feat.get('tls_sni') or 'N/A'}`")
                    st.markdown(f"**TLS Version:** `{feat.get('tls_version') or 'N/A'}`")
                    st.markdown(f"**JA3 Fingerprint:** `{feat.get('tls_ja3') or 'N/A'}`")
                    st.markdown(f"**Cipher Suites Count:** `{feat.get('tls_cipher_suites_count') or 0}`")
                    if feat.get("beacon_gap_mean") is not None:
                        st.markdown(
                            f"**Beacon Gap Mean:** `{feat.get('beacon_gap_mean'):.2f}s` (CV: `{feat.get('beacon_gap_cv'):.2f}`)"
                        )

                    st.markdown("<br>", unsafe_allow_html=True)
                    use_llm_flag = st.checkbox("Enable LLM Investigation", value=False, help="Requires OPENAI_API_KEY")
                    if st.button("🚀 Run Deep Investigation", key=f"inv_btn_{selected_flow_id}"):
                        with st.spinner("Executing offline tools and analysis..."):
                            try:
                                inv_res = api_post(
                                    st.session_state.api_url,
                                    "/investigate",
                                    {"flow_id": selected_flow_id, "use_llm": use_llm_flag},
                                )
                                st.success("Investigation complete! Check the Investigations tab.")
                                st.json(inv_res.get("report") or inv_res)
                            except Exception as e:
                                st.error(f"Investigation failed: {e}")

        # Scatter plot: bytes/s vs packets/s
        st.markdown("---")
        st.markdown("### Traffic Dynamics: Transfer Rates")
        scatter_data = []
        for fl in flow_records:
            pred = fl.get("prediction") or {}
            dur = max(float(fl.get("duration") or 0.0), 0.001)
            b = float(fl.get("bytes") or 0.0)
            p = float(fl.get("packets") or 0.0)
            scatter_data.append(
                {
                    "flow_id": fl.get("flow_id"),
                    "bytes_per_second": b / dur,
                    "packets_per_second": p / dur,
                    "label": pred.get("label", "unscored"),
                    "src": fl.get("src_ip"),
                    "dst": f"{fl.get('dst_ip')}:{fl.get('dst_port')}",
                }
            )
        if scatter_data:
            df_scatter = pd.DataFrame(scatter_data)
            fig_scatter = px.scatter(
                df_scatter,
                x="bytes_per_second",
                y="packets_per_second",
                color="label",
                color_discrete_map={"suspicious": COLOR_SUSPICIOUS, "benign": COLOR_BENIGN},
                hover_data=["flow_id", "src", "dst"],
                title="Flow Rate Distribution (Bytes/sec vs Packets/sec)",
                labels={"bytes_per_second": "Bytes / Second", "packets_per_second": "Packets / Second"},
            )
            apply_dark_theme(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# TAB 3: NETWORK
# ==============================================================================
with tab_network:
    st.subheader("Network Topology & Interaction Graph")
    top_n = st.slider("Limit to Top N Communicating Hosts", min_value=5, max_value=50, value=20)

    try:
        network_flows = api_get(st.session_state.api_url, "/flows", (("limit", 200),))
    except Exception as exc:
        st.error(f"Error fetching network flows: {exc}")
        network_flows = []

    if network_flows:
        # Build directed graph
        G = nx.DiGraph()
        flow_counts: dict[str, int] = {}
        edge_data: dict[tuple[str, str], dict[str, float]] = {}

        for fl in network_flows:
            src = fl.get("src_ip")
            dst = fl.get("dst_ip")
            if not src or not dst:
                continue
            pred = fl.get("prediction") or {}
            is_susp = 1.0 if pred.get("label") == "suspicious" else 0.0

            flow_counts[src] = flow_counts.get(src, 0) + 1
            flow_counts[dst] = flow_counts.get(dst, 0) + 1

            pair = (src, dst)
            if pair not in edge_data:
                edge_data[pair] = {"count": 0, "susp_count": 0}
            edge_data[pair]["count"] += 1
            edge_data[pair]["susp_count"] += is_susp

        # Filter top N hosts
        sorted_hosts = sorted(flow_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        valid_hosts = {h for h, _ in sorted_hosts}

        for (u, v), d in edge_data.items():
            if u in valid_hosts and v in valid_hosts:
                G.add_edge(u, v, weight=d["count"], susp_ratio=d["susp_count"] / max(d["count"], 1))

        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42)

            # Build edges traces
            edge_traces = []
            for u, v, data in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                susp_ratio = data.get("susp_ratio", 0.0)
                # Interpolate color from green to red based on susp_ratio
                edge_color = COLOR_SUSPICIOUS if susp_ratio > 0.0 else "rgba(100, 116, 139, 0.5)"
                width = min(max(1.0, math.sqrt(data.get("weight", 1))), 8.0)
                edge_traces.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode="lines",
                        line=dict(width=width, color=edge_color),
                        hoverinfo="text",
                        text=f"{u} → {v} | Flows: {data.get('weight')} | Suspicious: {susp_ratio*100:.0f}%",
                    )
                )

            # Node trace
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                cnt = flow_counts.get(node, 1)
                node_text.append(f"Host: {node}<br>Flows: {cnt}")
                node_size.append(min(max(10, cnt * 3), 35))

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[str(n) for n in G.nodes()],
                textposition="bottom center",
                hoverinfo="text",
                hovertext=node_text,
                marker=dict(
                    color=COLOR_ACCENT,
                    size=node_size,
                    line=dict(width=2, color="#ffffff"),
                ),
            )

            fig_graph = go.Figure(data=[*edge_traces, node_trace])
            fig_graph.update_layout(
                title="Host Communication Graph (Red edges = Suspicious Traffic)",
                showlegend=False,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            apply_dark_theme(fig_graph)
            st.plotly_chart(fig_graph, use_container_width=True)

        # Sankey Diagram: Source IP -> Destination Port -> Label
        st.markdown("---")
        st.markdown("### Traffic Flow Hierarchy (Sankey Diagram)")
        sankey_flows = network_flows[:100]
        if sankey_flows:
            labels_set = []
            for fl in sankey_flows:
                pred = fl.get("prediction") or {}
                src = fl.get("src_ip", "unknown")
                port = f"Port {fl.get('dst_port', '0')}"
                lbl = pred.get("label", "unscored")
                for item in [src, port, lbl]:
                    if item not in labels_set:
                        labels_set.append(item)

            label_map = {name: idx for idx, name in enumerate(labels_set)}
            link_counts: dict[tuple[int, int], int] = {}

            for fl in sankey_flows:
                pred = fl.get("prediction") or {}
                src_idx = label_map[fl.get("src_ip", "unknown")]
                port_idx = label_map[f"Port {fl.get('dst_port', '0')}"]
                lbl_idx = label_map[pred.get("label", "unscored")]

                link_counts[(src_idx, port_idx)] = link_counts.get((src_idx, port_idx), 0) + 1
                link_counts[(port_idx, lbl_idx)] = link_counts.get((port_idx, lbl_idx), 0) + 1

            sources = [k[0] for k in link_counts.keys()]
            targets = [k[1] for k in link_counts.keys()]
            values = list(link_counts.values())

            fig_sankey = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=labels_set,
                            color=[
                                COLOR_SUSPICIOUS if l == "suspicious" else (COLOR_BENIGN if l == "benign" else COLOR_ACCENT)
                                for l in labels_set
                            ],
                        ),
                        link=dict(source=sources, target=targets, value=values, color="rgba(100,116,139,0.3)"),
                    )
                ]
            )
            fig_sankey.update_layout(title="Source IP → Port → Threat Label Path")
            apply_dark_theme(fig_sankey)
            st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("No flow data available to render network interactions.")


# ==============================================================================
# TAB 4: INVESTIGATIONS
# ==============================================================================
with tab_investigations:
    st.subheader("Autonomous & Deterministic Investigation Dossiers")
    try:
        investigations = api_get(st.session_state.api_url, "/investigations")
    except Exception as exc:
        st.error(f"Error fetching investigations: {exc}")
        investigations = []

    if not investigations:
        st.info("No investigation dossiers compiled yet. Select a flow in the Flows tab to launch an investigation.")
    else:
        for inv in investigations:
            inv_id = inv.get("investigation_id")
            flow_id = inv.get("flow_id")
            status = inv.get("status")
            model_used = inv.get("model", "offline-tool-pass")
            created_at = inv.get("created_at", "")
            is_llm = "gpt" in model_used.lower()

            badge_mode = "🤖 LLM Agent" if is_llm else "⚙️ Offline Tool Pass"
            with st.expander(f"📁 {inv_id} | Flow: {flow_id} | Status: {status.upper()} ({badge_mode})"):
                st.caption(f"Compiled: {created_at} | Engine: {model_used} | Status: {status}")

                report = inv.get("report") or {}
                if isinstance(report, str):
                    try:
                        report = json.loads(report)
                    except Exception:
                        report = {"raw": report}

                if status == "failed":
                    st.error(f"Investigation execution failed: {report.get('error', 'Unknown failure')}")
                else:
                    # Render structured sections
                    c_sum, c_ev = st.columns(2)
                    with c_sum:
                        st.markdown("#### 📝 Executive Summary")
                        summary_text = report.get("summary") or report.get("executive_summary") or "No summary recorded."
                        st.write(summary_text)

                    with c_ev:
                        st.markdown("#### 🔬 Evidence & Indicators")
                        evidence = report.get("evidence") or []
                        if isinstance(evidence, list) and evidence:
                            for ev in evidence:
                                st.markdown(f"- {ev}")
                        elif isinstance(evidence, dict):
                            for k, v in evidence.items():
                                st.markdown(f"- **{k}**: {v}")
                        else:
                            st.write(str(evidence) if evidence else "No primary evidence listed.")

                    st.markdown("---")
                    c_rules, c_alt = st.columns(2)
                    with c_rules:
                        st.markdown("#### ⚖️ Matched Rules & Policies")
                        rules = report.get("rule_matches") or report.get("matched_rules") or []
                        if isinstance(rules, list) and rules:
                            for r in rules:
                                if isinstance(r, dict):
                                    st.markdown(f"- **{r.get('rule_id')}**: {r.get('description', '')}")
                                else:
                                    st.markdown(f"- {r}")
                        else:
                            st.markdown("*No specific detection rule triggers recorded.*")

                    with c_alt:
                        st.markdown("#### 🔄 Alternative Explanations")
                        alts = report.get("alternative_explanations") or report.get("alternatives") or []
                        if isinstance(alts, list) and alts:
                            for a in alts:
                                st.markdown(f"- {a}")
                        else:
                            st.markdown("*None noted.*")

                    st.markdown("---")
                    c_lim, c_next = st.columns(2)
                    with c_lim:
                        st.markdown("#### ⚠️ Operational Limitations")
                        lims = report.get("limitations") or []
                        if isinstance(lims, list) and lims:
                            for l in lims:
                                st.markdown(f"- {l}")
                        else:
                            st.markdown(str(lims) if lims else "*No explicit limitations logged.*")

                    with c_next:
                        st.markdown("#### 🎯 Recommended Next Steps")
                        steps = report.get("next_steps") or []
                        if isinstance(steps, list) and steps:
                            for s in steps:
                                st.markdown(f"- {s}")
                        else:
                            st.markdown(str(steps) if steps else "*Standard triage protocol.*")

                # Audit trail
                tool_calls = inv.get("tool_calls") or inv.get("tool_calls_json") or []
                if isinstance(tool_calls, str):
                    try:
                        tool_calls = json.loads(tool_calls)
                    except Exception:
                        tool_calls = []

                with st.expander("🛠️ Tool-Call Audit Trail"):
                    if tool_calls:
                        st.json(tool_calls)
                    else:
                        st.info("No tool calls recorded for this investigation.")


# ==============================================================================
# TAB 5: ACTIONS
# ==============================================================================
with tab_actions:
    st.subheader("Data Ingestion, Model Management & Evaluation")

    col_ingest, col_ml = st.columns(2)

    with col_ingest:
        st.markdown("### 📤 PCAP File Ingestion")

        # File upload
        uploaded_file = st.file_uploader("Upload PCAP capture (.pcap, .pcapng)", type=["pcap", "pcapng"])
        if uploaded_file is not None:
            if st.button("Extract Uploaded PCAP", use_container_width=True):
                with st.spinner("Uploading and dissecting packet metadata..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                        upload_res = api_upload(st.session_state.api_url, "/captures/upload", files)
                        st.success(f"Dissected {upload_res.get('flow_count')} flows (Capture ID: {upload_res.get('capture_id')})")
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Upload and extraction failed: {e}")

        st.markdown("---")
        st.markdown("### 📂 Extract Local PCAP")
        local_pcap = st.text_input("Server Path (restricted to data/samples or data/uploads)", value="data/samples/sample.pcap")
        if st.button("Extract Local PCAP", use_container_width=True):
            with st.spinner("Dissecting local PCAP..."):
                try:
                    res = api_post(st.session_state.api_url, "/captures/extract", params={"pcap_path": local_pcap})
                    st.success(f"Extracted {res.get('flow_count')} flows (Capture ID: {res.get('capture_id')})")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

    with col_ml:
        st.markdown("### 🧠 Machine Learning Engine")

        st.markdown("#### 1. Train Model")
        model_mode = st.selectbox("Training Algorithm", ["isolation_forest", "random_forest"])
        if st.button("Train Model on Ingested Data", use_container_width=True):
            with st.spinner("Training model artifact..."):
                try:
                    train_res = api_post(st.session_state.api_url, "/train", {"mode": model_mode})
                    st.success(
                        f"Trained {train_res.get('mode')} on {train_res.get('rows_trained')} flows. "
                        f"Threshold: {train_res.get('score_threshold', 'N/A')}"
                    )
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Training failed: {e}")

        st.markdown("---")
        st.markdown("#### 2. Run Anomaly Detection")
        if st.button("Score Flows with Saved Model", use_container_width=True):
            with st.spinner("Scoring flows..."):
                try:
                    detect_res = api_post(st.session_state.api_url, "/detect", {})
                    st.success(
                        f"Scored {detect_res.get('flows_scored')} flows. "
                        f"Suspicious: {detect_res.get('suspicious')} | Benign: {detect_res.get('benign')}"
                    )
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Detection scoring failed: {e}")
