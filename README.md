# Encrypted Traffic Threat Detection (ETTD)

Detect suspicious network communication from **encrypted traffic metadata** without decrypting application payloads. Suspicious flows can be investigated by an active **Threat Investigation Agent** that calls read-only Python analysis tools and produces an evidence-backed report.

## Security Constraints

- **Offline authorized PCAPs only** — do not capture traffic without authorization.
- **No decryption** of TLS or other application payloads.
- **No automatic blocking** or firewall changes.
- Agent tools are **read-only** and argument-validated (no shell access).
- API keys live in environment variables (see `.env.example`); never commit `.env`.
- Packet contents, logs, and metadata are **untrusted input** and must not override agent instructions.

## Architecture

```
PCAP → ingestion (TLS/JA3) → flow builder (initiator direction) → feature extraction → SQLite
                                                                         ↓
                                                    Separate Train (/train) & Score (/detect)
                                                      (Isolation Forest / Random Forest)
                                                                         ↓
                                                            Threat Investigation Agent
                                                              + read-only tools
                                                                         ↓
                                                        FastAPI Backend ← → Streamlit UI
```

Package layout lives under `src/ettd/` with separate modules for ingestion, flows, features, ML, agent, DB, and API. The Streamlit UI is in `dashboard/app.py`.

## Quick Start

```powershell
# 1. Environment setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# 2. Generate rich demo synthetic PCAP (with real TLS ClientHellos & periodic beaconing)
python scripts\generate_sample_pcap.py

# 3. Extract flow metadata & features into SQLite
ettd-extract data\samples\sample.pcap --out data\processed\flows.csv

# 4. Train anomaly detection model
# Or use python scripts\run_detect.py
ettd-detect

# 5. Run tests (all passing)
pytest -q

# 6. End-to-end pipeline (extract -> detect -> investigate offline)
ettd-pipeline

# 7. Start FastAPI backend
uvicorn ettd.api.main:app --reload --app-dir src

# 8. Start Streamlit dashboard (in a separate terminal)
streamlit run dashboard\app.py
```

### Optional LLM Investigation (Requires OpenAI API Key)

```powershell
$env:OPENAI_API_KEY="sk-..."
python scripts\run_pipeline.py --use-llm
```

Without `OPENAI_API_KEY`, LLM investigation fails gracefully with `status="failed"` and clear error notes rather than returning unhandled 500s or hanging in `pending`. Offline tool-backed investigation is always available via `use_llm=False`.

## Dashboard Overview & Visuals

The redesigned Streamlit dashboard (`dashboard/app.py`) features a dark theme and 5 tabs:

1. **📊 Overview**:
   - Real-time KPI summary (Flows analyzed, Suspicious flows, Benign flows, Investigations).
   - Anomaly score distribution histogram with trained threshold indicator.
   - Flow volume & threat timeline stacked bar chart.
   - Top 10 destination talkers (by flow count and by byte volume).
   - Protocol and destination port share donut chart.
2. **🌊 Flows**:
   - Filterable, sortable flow catalog with color-coded threat labels.
   - Server-side pagination, search by IP/SNI, port filter, and min score threshold.
   - Selected flow inspector: feature-deviation explanation bar chart (pure numeric z-score / importance metrics), TLS metadata & JA3 hashes, related flows, and one-click investigation trigger.
   - Traffic dynamics scatter plot (`bytes_per_second` vs `packets_per_second`).
3. **🕸️ Network**:
   - Interactive source-to-destination host communication graph (directed edges, red highlighting for suspicious links).
   - Host slider to limit visualization to top communicating nodes.
   - Traffic hierarchy Sankey diagram (`Source IP → Destination Port → Threat Label`).
4. **🔍 Investigations**:
   - Structured investigation dossiers (Executive Summary, Evidence, Matched Rules, Alternative Explanations, Limitations, Next Steps).
   - Distinct tags for Offline Tool Pass vs LLM Agent execution.
   - Full tool-call audit trail expander.
5. **⚡ Actions**:
   - Browser PCAP upload (`/captures/upload`) with file-size limits and unique timestamped storage.
   - Local PCAP extraction with directory path restriction and traversal protection.
   - Model retraining and inference controls.

## ML Modes & Separation

| Mode | When | Scoring Behavior |
|---|---|---|
| **Isolation Forest** | Default (unsupervised) | Raw decision function against trained threshold stored in `model_meta.json`. Scores are relative anomaly rankings. |
| **Random Forest** | When `labels.path` points to a validated CSV (`flow_id`, `label`) | Supervised classifier score. Training rows are explicitly tagged (`is_training_row=True`) and never presented as test results. |

Training and scoring are strictly decoupled:
- `POST /train`: Trains a versioned model in `data/models/`, creates a `latest.json` pointer, and returns training metrics.
- `POST /detect`: Scores flows using the loaded model artifact; cleanly errors if no model has been trained.
- `GET /flows/{id}/explain`: Returns the top 5 deviant features (absolute z-scores for Isolation Forest, importance × deviation for Random Forest) without LLM hallucinations.

## Threat Intelligence & JA3 Fingerprinting

- ClientHello packets extract real TLS attributes: SNI, TLS version, cipher suite lists, and MD5 JA3 fingerprints.
- `config/ja3_blocklist.txt` provides local, offline threat intelligence matching without any external network calls.
- Detection rules in `config/detection_rules.yaml` evaluate heuristics such as periodic beaconing (`beacon_gap_cv < 0.25`), high outbound asymmetry, repeated destination bursts, and missing TLS on port 443.

## Known Limitations

- **Offline Metadata Only:** ETTD analyzes headers, packet sizes, and handshake metadata; encrypted payloads are never inspected or decrypted.
- **Handshake Completeness:** Captures missing the initial SYN or ClientHello cannot extract SNI or JA3 hashes (explicitly marked with missing indicators).
- **Anomaly Scores vs Probabilities:** Isolation Forest anomaly scores reflect geometric distance in the feature space, not calibrated threat probabilities.
- **Heuristic Threat Rules:** Behavioral rules are transparent heuristics to guide analyst triage, not definitive proof of compromise.
- **Local Threat Intelligence:** Threat intel is restricted to local offline blocklists (`config/ja3_blocklist.txt`); no live external lookups are performed.
- **Lab & Educational Scope:** Designed for authorized security research, CTFs, and offline incident triage—not an inline production IPS.
