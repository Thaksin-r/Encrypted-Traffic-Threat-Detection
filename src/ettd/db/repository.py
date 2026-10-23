"""SQLite persistence for captures, flows, features, predictions, investigations."""

from __future__ import annotations

import contextlib
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

    def _connect(self) -> contextlib.AbstractContextManager[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return contextlib.closing(conn)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def create_capture(self, source_path: str, capture_id: str | None = None) -> str:
        cid = capture_id or str(uuid.uuid4())
        with self._connect() as conn:
            with conn:
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
        flow_rows = []
        feature_rows = []
        cols = list(features_df.columns)
        records = features_df.to_dict(orient="records")

        for row in records:
            flow_id = str(row["flow_id"])
            start_t, end_t = flow_times.get(flow_id, (None, None))
            flow_rows.append(
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
                )
            )
            payload = {col: _jsonable(row[col]) for col in cols}
            feature_rows.append((flow_id, capture_id, json.dumps(payload)))

        with self._connect() as conn:
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO flows
                    (flow_id, capture_id, flow_key, src_ip, dst_ip, src_port, dst_port, protocol, start_time, end_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    flow_rows,
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO features (flow_id, capture_id, feature_json) VALUES (?, ?, ?)",
                    feature_rows,
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

    def get_flow(self, flow_id: str) -> dict[str, Any] | None:
        """Fetch a single flow by flow_id directly via SQL without full table scans."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM flows WHERE flow_id = ?", (flow_id,)).fetchone()
        return dict(row) if row else None

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

    def list_flows_enriched(
        self,
        capture_id: str | None = None,
        label: str | None = None,
        dst_ip: str | None = None,
        dst_port: int | None = None,
        min_score: float | None = None,
        search: str | None = None,
        sort_by: str = "flow_id",
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Fetch flows joined with features and predictions in a single query.

        Supports pagination (limit, offset), score sorting, and filtering by label/ip/port/min_score.
        """
        where_clauses: list[str] = []
        params: list[Any] = []

        if capture_id:
            where_clauses.append("f.capture_id = ?")
            params.append(capture_id)
        if label:
            where_clauses.append("p.label = ?")
            params.append(label)
        if dst_ip:
            where_clauses.append("f.dst_ip = ?")
            params.append(dst_ip)
        if dst_port is not None:
            where_clauses.append("f.dst_port = ?")
            params.append(int(dst_port))
        if min_score is not None:
            where_clauses.append("p.score >= ?")
            params.append(float(min_score))
        if search:
            search_pattern = f"%{search.strip()}%"
            where_clauses.append("(f.src_ip LIKE ? OR f.dst_ip LIKE ? OR fe.feature_json LIKE ?)")
            params.extend([search_pattern, search_pattern, search_pattern])

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        order_dir = "DESC" if sort_order.lower() == "desc" else "ASC"
        if sort_by == "score":
            order_sql = f" ORDER BY (p.score IS NULL), p.score {order_dir}, f.flow_id ASC"
        elif sort_by == "start_time":
            order_sql = f" ORDER BY (f.start_time IS NULL), f.start_time {order_dir}, f.flow_id ASC"
        elif sort_by == "dst_port":
            order_sql = f" ORDER BY f.dst_port {order_dir}, f.flow_id ASC"
        else:
            order_sql = f" ORDER BY f.flow_id {order_dir}"

        pagination_sql = ""
        if limit is not None:
            pagination_sql = " LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])

        query = f"""
        SELECT
            f.flow_id,
            f.capture_id,
            f.flow_key,
            f.src_ip,
            f.dst_ip,
            f.src_port,
            f.dst_port,
            f.protocol,
            f.start_time,
            f.end_time,
            fe.feature_json,
            p.label,
            p.score,
            p.model_mode,
            p.created_at AS prediction_created_at
        FROM flows f
        LEFT JOIN features fe ON f.flow_id = fe.flow_id
        LEFT JOIN predictions p ON f.flow_id = p.flow_id
        {where_sql}
        {order_sql}
        {pagination_sql}
        """

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        enriched = []
        for r in rows:
            feat = json.loads(r["feature_json"]) if r["feature_json"] else {}
            pred = None
            if r["label"] is not None:
                pred = {
                    "flow_id": r["flow_id"],
                    "capture_id": r["capture_id"],
                    "label": r["label"],
                    "score": float(r["score"]),
                    "model_mode": r["model_mode"],
                    "created_at": r["prediction_created_at"],
                }
            enriched.append(
                {
                    "flow_id": r["flow_id"],
                    "capture_id": r["capture_id"],
                    "flow_key": r["flow_key"],
                    "src_ip": r["src_ip"],
                    "dst_ip": r["dst_ip"],
                    "src_port": r["src_port"],
                    "dst_port": r["dst_port"],
                    "protocol": r["protocol"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "features": feat,
                    "prediction": pred,
                }
            )
        return enriched

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
        now = _utc_now()
        pred_rows = [
            (
                pred["flow_id"],
                capture_id,
                pred["label"],
                float(pred["score"]),
                pred["model_mode"],
                now,
            )
            for pred in predictions
        ]
        with self._connect() as conn:
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO predictions
                    (flow_id, capture_id, label, score, model_mode, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    pred_rows,
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
            with conn:
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
            with conn:
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

    def get_score_distribution(self, num_bins: int = 10) -> dict[str, Any]:
        """Histogram data of prediction scores."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT score, label FROM predictions WHERE score IS NOT NULL"
            ).fetchall()
        if not rows:
            return {
                "bin_edges": [],
                "bin_labels": [],
                "counts": [],
                "suspicious_counts": [],
                "total": 0,
            }
        scores = [float(r["score"]) for r in rows]
        is_susp = [1 if r["label"] == "suspicious" else 0 for r in rows]

        min_s = min(scores)
        max_s = max(scores)
        low = 0.0 if min_s >= 0.0 and max_s <= 1.0 else min_s
        high = 1.0 if min_s >= 0.0 and max_s <= 1.0 else max_s
        span = max(1e-6, high - low)

        bin_width = span / num_bins
        bin_edges = [round(low + i * bin_width, 3) for i in range(num_bins + 1)]
        counts = [0] * num_bins
        susp_counts = [0] * num_bins

        for s, susp in zip(scores, is_susp):
            idx = int((s - low) / bin_width)
            if idx >= num_bins:
                idx = num_bins - 1
            if idx < 0:
                idx = 0
            counts[idx] += 1
            if susp:
                susp_counts[idx] += 1

        bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(num_bins)]
        return {
            "bin_edges": bin_edges,
            "bin_labels": bin_labels,
            "counts": counts,
            "suspicious_counts": susp_counts,
            "total": len(scores),
        }

    def get_timeline(self, bucket_seconds: float = 60.0) -> list[dict[str, Any]]:
        """Flows and suspicious flows aggregated per time bucket."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.start_time, p.label
                FROM flows f
                LEFT JOIN predictions p ON f.flow_id = p.flow_id
                WHERE f.start_time IS NOT NULL
                ORDER BY f.start_time ASC
                """
            ).fetchall()
        if not rows:
            return []

        times = [float(r["start_time"]) for r in rows]
        min_t = min(times)
        max_t = max(times)
        span = max_t - min_t

        if span <= 0:
            bucket_sz = 1.0
        elif span / bucket_seconds > 40:
            bucket_sz = max(0.001, span / 30.0)
        elif span / bucket_seconds < 4:
            bucket_sz = max(0.001, span / 10.0)
        else:
            bucket_sz = bucket_seconds

        buckets: dict[float, dict[str, Any]] = {}
        for r in rows:
            st = float(r["start_time"])
            b_key = min_t + (int((st - min_t) / bucket_sz) * bucket_sz)
            b_key = round(b_key, 3)
            if b_key not in buckets:
                buckets[b_key] = {"time": b_key, "flow_count": 0, "suspicious_count": 0}
            buckets[b_key]["flow_count"] += 1
            if r["label"] == "suspicious":
                buckets[b_key]["suspicious_count"] += 1

        return sorted(buckets.values(), key=lambda b: b["time"])

    def get_top_talkers(self, limit: int = 10) -> dict[str, Any]:
        """Aggregated top talkers for dashboard charts."""
        with self._connect() as conn:
            dest_flows = conn.execute(
                """
                SELECT f.dst_ip, COUNT(*) AS flow_count,
                       SUM(CASE WHEN p.label = 'suspicious' THEN 1 ELSE 0 END) AS suspicious_count
                FROM flows f
                LEFT JOIN predictions p ON f.flow_id = p.flow_id
                GROUP BY f.dst_ip
                ORDER BY flow_count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            dest_ports = conn.execute(
                """
                SELECT dst_port, COUNT(*) AS count
                FROM flows
                GROUP BY dst_port
                ORDER BY count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            protocols = conn.execute(
                """
                SELECT protocol, COUNT(*) AS count
                FROM flows
                GROUP BY protocol
                ORDER BY count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            feat_rows = conn.execute(
                "SELECT feature_json FROM features WHERE feature_json IS NOT NULL"
            ).fetchall()

        bytes_by_dst: dict[str, int] = {}
        for r in feat_rows:
            try:
                data = json.loads(r["feature_json"])
                dst = data.get("dst_ip")
                b = int(data.get("total_bytes") or 0)
                if dst:
                    bytes_by_dst[dst] = bytes_by_dst.get(dst, 0) + b
            except Exception:
                continue

        top_bytes = sorted(
            [{"dst_ip": k, "total_bytes": v} for k, v in bytes_by_dst.items()],
            key=lambda x: x["total_bytes"],
            reverse=True,
        )[:limit]

        return {
            "top_destinations_by_flows": [dict(r) for r in dest_flows],
            "top_destinations_by_bytes": top_bytes,
            "top_ports": [dict(r) for r in dest_ports],
            "protocols": [dict(r) for r in protocols],
        }


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value
