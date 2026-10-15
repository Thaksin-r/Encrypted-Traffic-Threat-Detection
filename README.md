# Encrypted Traffic Threat Detection (ETTD)

Detect suspicious network communication from **encrypted traffic metadata** without decrypting application payloads. Suspicious flows can be investigated by an active **Threat Investigation Agent** that calls read-only Python analysis tools and produces an evidence-backed report.

## Security constraints

- **Offline authorized PCAPs only** — do not capture traffic without authorization.
- **No decryption** of TLS or other application payloads.
- **No automatic blocking** or firewall changes.
- Agent tools are **read-only** and argument-validated (no shell access).
- API keys live in environment variables (see `.env.example`); never commit `.env`.
- Packet contents, logs, and metadata are **untrusted input** and must not override agent instructions.

## Architecture

```
PCAP → ingestion → flow builder → feature extraction → SQLite
                                      ↓
                               ML detection
                          (Isolation Forest default,
                           Random Forest if labels valid)
                                      ↓
                         Threat Investigation Agent
                           + read-only tools
                                      ↓
                         FastAPI ← → Streamlit dashboard
```

Package layout lives under `src/ettd/` with separate modules for ingestion, flows, features, ML, agent, DB, and API. The Streamlit UI is in `dashboard/`.

## Quick start

```powershell
cd C:\Users\Thaksin\Projects\encrypted-traffic-threat-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# Generate synthetic sample PCAP (lab data only)
python scripts\generate_sample_pcap.py

# Extract flow features
python -m ettd.pipeline_extract data\samples\sample.pcap --out data\processed\flows.csv

# Run tests
pytest -q

# End-to-end offline pipeline (tools + deterministic report, no LLM)
python scripts\run_pipeline.py

# API
uvicorn ettd.api.main:app --reload --app-dir src

# Dashboard (separate terminal)
streamlit run dashboard\app.py
```

Optional LLM investigation (requires key):

```powershell
$env:OPENAI_API_KEY="sk-..."
python scripts\run_pipeline.py --use-llm
```

Without `OPENAI_API_KEY`, LLM investigation **fails clearly** and does not fabricate an agent report. Offline tool-backed investigation is available via `use_llm=False` (API/dashboard checkbox / tests).

## ML modes

| Mode | When |
|---|---|
| **Isolation Forest** | Default when no valid labels file is configured |
| **Random Forest** | When `labels.path` / `ETTD_LABELS_PATH` points to a valid CSV (`flow_id`, `label`) |

Scores are **ranking signals**, not calibrated probabilities, unless calibration is separately evaluated.

See `data/labels/README.md` for the labels schema.

## Investigation tools

- `inspect_flow`
- `analyze_tls_metadata` (explicit `unavailable` when missing)
- `find_related_flows`
- `compare_detection_rules` (`config/detection_rules.yaml`)
- `check_threat_intelligence` (reports “no lookup performed” if no source configured)

## Limitations

- Metadata-only features cannot identify every encrypted threat.
- Incomplete PCAPs may omit handshake fields (e.g., SNI).
- Unsupervised anomaly scores reflect deviation in the current feature space, not confirmed malice.
- Behavioral rules are transparent heuristics, not ground truth.
- Threat-intelligence connectors are not implemented in the MVP; the tool refuses silent fabrication.
- This project is for learning and authorized lab analysis — not a production IDS replacement.

## Milestone map

1. PCAP → flows → features  
2. SQLite persistence  
3. Dual-mode ML detection  
4. Investigation tools  
5. OpenAI Agents SDK investigator  
6. FastAPI backend  
7. Streamlit dashboard  
8. Docs / hardening  

## License / ethics

Use only on traffic you are authorized to analyze. Do not deploy for unauthorized monitoring.
