"""SQLite persistence for captures, flows, features, predictions, investigations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ettd.db.models import SCHEMA_SQL


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def create_capture(self, source_path: str, capture_id: str | None = None) -> str:
        cid = capture_id or str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO captures (capture_id, source_path, created_at, flow_count) VALUES (?, ?, ?, 0)",
                (cid, source_path, _utc_now()),
            )
        return cid

    def save_flows_and_features(
        self,
        capture_id: str,
        features_df: pd.DataFrame,
        flow_times: dict[str, tuple[float, float]] | None = None,
    ) -> int:
        flow_times = flow_times or {}
        with self._connect() as conn:
            for _, row in features_df.iterrows():
                flow_id = str(row["flow_id"])
                start_t, end_t = flow_times.get(flow_id, (None, None))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO flows
                    (flow_id, capture_id, flow_key, src_ip, dst_ip, src_port, dst_port, protocol, start_time, end_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        flow_id,
                        capture_id,
                        str(row["flow_key"]),
                        str(row["src_ip"]),
                        str(row["dst_ip"]),
                        int(row["src_port"]),
                        int(row["dst_port"]),
                        str(row["protocol"]),
                        start_t,
                        end_t,
                    ),
                )
                payload = {col: _jsonable(row[col]) for col in features_df.columns}
                conn.execute(
                    "INSERT OR REPLACE INTO features (flow_id, capture_id, feature_json) VALUES (?, ?, ?)",
                    (flow_id, capture_id, json.dumps(payload)),
                )
            conn.execute(
                "UPDATE captures SET flow_count = ? WHERE capture_id = ?",
                (len(features_df), capture_id),
            )
        return len(features_df)

    def list_captures(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM captures ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def list_flows(self, capture_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if capture_id:
                rows = conn.execute(
                    "SELECT * FROM flows WHERE capture_id = ? ORDER BY flow_id",
                    (capture_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM flows ORDER BY flow_id").fetchall()
        return [dict(r) for r in rows]

    def get_flow_features(self, flow_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT feature_json FROM features WHERE flow_id = ?",
                (flow_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["feature_json"])

    def get_features_dataframe(self, capture_id: str | None = None) -> pd.DataFrame:
        with self._connect() as conn:
            if capture_id:
                rows = conn.execute(
                    "SELECT feature_json FROM features WHERE capture_id = ?",
                    (capture_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT feature_json FROM features").fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([json.loads(r["feature_json"]) for r in rows])

    def save_predictions(self, capture_id: str, predictions: list[dict[str, Any]]) -> None:
        with self._connect() as conn:
            for pred in predictions:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO predictions
                    (flow_id, capture_id, label, score, model_mode, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pred["flow_id"],
                        capture_id,
                        pred["label"],
                        float(pred["score"]),
                        pred["model_mode"],
                        _utc_now(),
                    ),
                )

    def get_prediction(self, flow_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE flow_id = ?",
                (flow_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_predictions(self, capture_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if capture_id:
                rows = conn.execute(
                    "SELECT * FROM predictions WHERE capture_id = ?",
                    (capture_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM predictions").fetchall()
        return [dict(r) for r in rows]

    def create_investigation(self, flow_id: str, capture_id: str) -> str:
        iid = str(uuid.uuid4())
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO investigations
                (investigation_id, flow_id, capture_id, status, report_json, tool_calls_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (iid, flow_id, capture_id, "pending", now, now),
            )
        return iid

    def update_investigation(
        self,
        investigation_id: str,
        *,
        status: str,
        report: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE investigations
                SET status = ?, report_json = ?, tool_calls_json = ?, updated_at = ?
                WHERE investigation_id = ?
                """,
                (
                    status,
                    json.dumps(report) if report is not None else None,
                    json.dumps(tool_calls) if tool_calls is not None else None,
                    _utc_now(),
                    investigation_id,
                ),
            )

    def get_investigation(self, investigation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM investigations WHERE investigation_id = ?",
                (investigation_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        if data.get("report_json"):
            data["report"] = json.loads(data["report_json"])
        if data.get("tool_calls_json"):
            data["tool_calls"] = json.loads(data["tool_calls_json"])
        return data

    def list_investigations(self, flow_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if flow_id:
                rows = conn.execute(
                    "SELECT * FROM investigations WHERE flow_id = ? ORDER BY created_at DESC",
                    (flow_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM investigations ORDER BY created_at DESC"
                ).fetchall()
        results = []
        for row in rows:
            data = dict(row)
            if data.get("report_json"):
                data["report"] = json.loads(data["report_json"])
            if data.get("tool_calls_json"):
                data["tool_calls"] = json.loads(data["tool_calls_json"])
            results.append(data)
        return results

    def summary_counts(self) -> dict[str, Any]:
        with self._connect() as conn:
            flows = conn.execute("SELECT COUNT(*) AS c FROM flows").fetchone()["c"]
            suspicious = conn.execute(
                "SELECT COUNT(*) AS c FROM predictions WHERE label = 'suspicious'"
            ).fetchone()["c"]
            benign = conn.execute(
                "SELECT COUNT(*) AS c FROM predictions WHERE label = 'benign'"
            ).fetchone()["c"]
            investigations = conn.execute("SELECT COUNT(*) AS c FROM investigations").fetchone()["c"]
        return {
            "flows_analyzed": flows,
            "suspicious": suspicious,
            "benign": benign,
            "investigations": investigations,
        }


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value
