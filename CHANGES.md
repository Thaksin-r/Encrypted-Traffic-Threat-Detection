# ETTD Changelog & Implementation Notes

This document summarizes the changes across all five parts of the ETTD (Encrypted Traffic Threat Detection) hardening and enhancement milestone.

---

## Part 1: Detection Bugs

1. **TLS Parsing Fix**:
   - **Problem**: `scapy.layers.tls` was imported inside `read_pcap()` after `rdpcap()` had already parsed packets as raw TCP data; non-standard ports (e.g., 8443) were ignored.
   - **Fix**: Executed `load_layer("tls")` at module import time in `src/ettd/ingestion/pcap_reader.py`. Added raw TCP payload inspection for TLS record header (`0x16 0x03`) to detect TLS handshakes regardless of port.
   - **Test**: Added `tests/test_tls_parsing.py` which builds ClientHello packets with SNI and cipher suites on ports 443 and 8443 and verifies extraction both in-process and in an isolated fresh process.

2. **Flow Direction Fix**:
   - **Problem**: `_canonical_key` used alphabetical IP ordering to define the "forward" direction, resulting in client-initiated flows having 0 forward packets and broken destination port metrics.
   - **Fix**: Redesigned `src/ettd/flows/builder.py` so the forward direction is strictly determined by the flow initiator (earliest packet timestamp, prioritizing TCP SYN without ACK). The sorted key is retained solely for grouping bidirectional packets.
   - **Test**: Added `test_flow_direction_both_orders` and `test_flow_direction_syn_priority` in `tests/test_flow_builder.py`.

3. **Isolation Forest Calibration & Over-Flagging**:
   - **Problem**: `infer.py` applied batch-level min-max normalization, compared against 0.55, and ORed with `-1`, invariably flagging the highest-scoring flow in any batch.
   - **Fix**: Removed batch min-max normalization. Used the model's raw `decision_function` evaluated against the threshold stored in `model_meta.json` during training, preserving consistent anomaly rankings.
   - **Test**: Added `test_isolation_forest_uniform_flows_contamination_level` in `tests/test_ml.py`.

4. **Training and Scoring Separation**:
   - **Problem**: `/detect` and `detect_suspicious(retrain=True)` continuously retrained on the scored data and overwrote a single shared model artifact.
   - **Fix**: Created dedicated `POST /train` endpoint and made `POST /detect` strictly inference-only against the loaded model. Saved versioned model artifacts with a `latest.json` pointer. Tagged Random Forest training rows (`is_training_row=True`) to prevent confusing training scores with test evaluations.

5. **LLM Investigation Robustness**:
   - **Problem**: `_llm_investigate` lacked exception handling, causing API requests to hang in `pending` or fail with unhandled 500 errors when API keys were missing or requests failed.
   - **Fix**: Wrapped `_llm_investigate` in `src/ettd/agent/investigator.py` with try/except, setting `status="failed"` and recording the error in the investigation report.
   - **Test**: Added `test_llm_investigate_failure_sets_status_failed` in `tests/test_tools.py`.

6. **Missing TLS Value Indicators**:
   - **Problem**: `fillna(0.0)` in `features_matrix.py` converted missing TLS attributes into deceptive zeros.
   - **Fix**: Added explicit indicator columns (e.g., `tls_cipher_suites_missing`) in `src/ettd/features/extractor.py` and `src/ettd/ml/features_matrix.py`.

---

## Part 2: Backend Hardening

1. **Path Traversal Protection**:
   - Restricted `/captures/extract` to allowed directories (`data/samples` and `data/uploads`). Rejected directory traversal patterns (`..`, absolute paths outside allowed roots) with HTTP 403 Forbidden.
2. **Secure Uploads**:
   - Implemented streaming file-size validation (50MB limit, HTTP 413 Payload Too Large) and unique timestamped filename generation in `POST /captures/upload`.
3. **Optimized Repository & Single-Query Enriched Flows**:
   - Added direct `get_flow(flow_id)` to eliminate full-table scanning in Python.
   - Replaced N+1 queries in `GET /flows` with a single SQL query joining `flows`, `flow_features`, and `predictions`.
   - Added server-side pagination (`limit`, `offset`), sorting (`sort_by`, `sort_order`), and filtering (`label`, `dst_ip`, `dst_port`, `min_score`, `search`).
4. **Dashboard Stats Endpoints**:
   - Added `GET /stats/score-distribution` (histogram binning and threshold).
   - Added `GET /stats/timeline` (time-bucketed flow volume and suspicious counts).
   - Added `GET /stats/top-talkers` (top destinations by flow count and bytes transferred, port distribution).
   - Added `GET /flows/{flow_id}/explain` (top 5 feature deviations by absolute z-score or importance-weighted deviation).
5. **Database Resource Management**:
   - Wrapped SQLite connections using `contextlib.closing()`. Replaced pandas `iterrows()` loops with batch `executemany()` for flow, feature, and prediction insertions.

---

## Part 3: Feature Additions

1. **Beaconing Features & Detection Rule**:
   - Implemented `beacon_gap_mean`, `beacon_gap_cv`, and `beacon_flow_count` per `(src_ip, dst_ip)` pair in `src/ettd/features/extractor.py`.
   - Added `beaconing_regular_timing` rule in `config/detection_rules.yaml` and implemented rule evaluation in `InvestigationTools`.
2. **JA3 Fingerprinting & Local Blocklist**:
   - Extracted MD5 JA3 fingerprints from TLS ClientHello packets and stored them as `tls_ja3`.
   - Created `config/ja3_blocklist.txt` and integrated local hash matching into `check_threat_intelligence` without external network lookups.
3. **Deterministic Explainability**:
   - Formulated `/flows/{id}/explain` to compute plain numeric metrics (z-scores against scaler mean/scale for Isolation Forest; feature importance × deviation for Random Forest) without LLM text generation.

---

## Part 4: Frontend (Streamlit Dashboard)

- Restructured `dashboard/app.py` into five distinct tabs using Plotly charts and a dark glassmorphism theme:
  1. **Overview**: KPI metrics, score distribution histogram with vertical threshold line, time-bucketed traffic & threat timeline, top 10 destinations by flow count and total bytes, protocol/port donut chart.
  2. **Flows**: Filterable, sortable dataframe with color-coded labels, server-side pagination, selected flow inspection panel with feature-deviation bar chart, TLS/JA3 metadata display, and one-click investigation trigger. Added flow rate scatter plot (bytes/s vs packets/s).
  3. **Network**: Interactive source-to-destination host communication graph (directed edges, red highlighting for suspicious links) with top N node slider, and a hierarchical Sankey diagram (`Source IP → Destination Port → Threat Label`).
  4. **Investigations**: Structured dossiers (Executive Summary, Evidence, Matched Rules, Alternatives, Limitations, Next Steps), distinct badges for Offline Tool Pass vs LLM Agent execution, and an expandable tool-call audit trail.
  5. **Actions**: Browser PCAP file uploader, server PCAP extraction, model retraining controls, and detection execution with progress indicators and error feedback.
- Added brief API response caching with `st.cache_data`.

---

## Part 5: Cleanup & Packaging

- **`pyproject.toml`**: Moved dependencies from `requirements.txt` into `[project.dependencies]`. Added console entry points: `ettd-extract`, `ettd-detect`, `ettd-pipeline`.
- **Scripts**: Removed `sys.path.insert` hacks in `scripts/run_detect.py`, `scripts/run_pipeline.py`, and `scripts/generate_sample_pcap.py`.
- **`README.md`**: Removed hardcoded absolute paths, added quick start guide, visual dashboard summary, and realistic known limitations list.
- **Configuration**: Gitignored `config/settings.yaml` and maintained `config/settings.example.yaml`.
- **Synthetic PCAP Generator**: Added `write_rich_sample_pcap` in `src/ettd/samples.py` and updated `scripts/generate_sample_pcap.py` to produce a rich test capture with real TLS ClientHellos and periodic beaconing hosts.

---

## What Was Intentionally Not Done & Why

1. **No External Network Lookups in Threat Intelligence**:
   - Kept threat intelligence strictly local via `config/ja3_blocklist.txt`. External API queries (e.g., VirusTotal, AbuseIPDB) were avoided to prevent accidental network egress, API quota dependencies, and data leakage from offline PCAP captures.
2. **No Automatic Firewall / Auto-Blocking Actions**:
   - Maintained human-in-the-loop triage by design. As per project constraints, detection results are advisory and investigative, preventing erroneous automated network disruptions.
3. **No Added Generative / LLM Features**:
   - All explainability, rule matching, beaconing detection, and graph analyses were kept strictly deterministic and mathematical.
