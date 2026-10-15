"""SQLite schema definitions."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS captures (
    capture_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    flow_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS flows (
    flow_id TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL,
    flow_key TEXT NOT NULL,
    src_ip TEXT NOT NULL,
    dst_ip TEXT NOT NULL,
    src_port INTEGER NOT NULL,
    dst_port INTEGER NOT NULL,
    protocol TEXT NOT NULL,
    start_time REAL,
    end_time REAL,
    FOREIGN KEY (capture_id) REFERENCES captures(capture_id)
);

CREATE TABLE IF NOT EXISTS features (
    flow_id TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL,
    feature_json TEXT NOT NULL,
    FOREIGN KEY (flow_id) REFERENCES flows(flow_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    flow_id TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    model_mode TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (flow_id) REFERENCES flows(flow_id)
);

CREATE TABLE IF NOT EXISTS investigations (
    investigation_id TEXT PRIMARY KEY,
    flow_id TEXT NOT NULL,
    capture_id TEXT NOT NULL,
    status TEXT NOT NULL,
    report_json TEXT,
    tool_calls_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (flow_id) REFERENCES flows(flow_id)
);
"""
