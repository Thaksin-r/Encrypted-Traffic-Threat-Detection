"""FastAPI backend for ETTD."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from ettd.agent.investigator import investigate_flow
from ettd.config import PROJECT_ROOT, db_path, load_settings
from ettd.db.repository import Repository
from ettd.ml.infer import _resolve_model_and_meta, detect_suspicious
from ettd.ml.train import train_model
from ettd.pipeline_extract import extract_from_pcap

app = FastAPI(
    title="Encrypted Traffic Threat Detection API",
    description="Metadata-only encrypted traffic analysis. No decryption. No auto-blocking.",
    version="0.1.0",
)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


def get_repo() -> Repository:
    return Repository(db_path(load_settings()))


def _validate_pcap_path(pcap_path: str, settings: dict[str, Any] | None = None) -> Path:
    """Validate PCAP path against allowed directories (data/samples, data/uploads) and reject traversal."""
    if not pcap_path or "\x00" in pcap_path:
        raise HTTPException(status_code=400, detail="Invalid PCAP path")

    p = Path(pcap_path)
    if not p.is_absolute() and len(p.parts) == 1:
        target = (PROJECT_ROOT / "data" / "samples" / p).resolve()
    elif not p.is_absolute():
        target = (PROJECT_ROOT / p).resolve()
    else:
        target = p.resolve()

    allowed_roots = [
        (PROJECT_ROOT / "data" / "samples").resolve(),
        (PROJECT_ROOT / "data" / "uploads").resolve(),
    ]
    if settings:
        for extra in settings.get("allowed_pcap_dirs", []):
            allowed_roots.append(Path(extra).resolve())
    import os
    env_extra = os.environ.get("ETTD_ALLOWED_PCAP_DIRS")
    if env_extra:
        for extra_p in env_extra.split(";"):
            if extra_p.strip():
                allowed_roots.append(Path(extra_p.strip()).resolve())

    is_allowed = False
    for root in allowed_roots:
        try:
            target.relative_to(root)
            is_allowed = True
            break
        except ValueError:
            pass

    if not is_allowed:
        raise HTTPException(
            status_code=403,
            detail="Access forbidden: Path outside allowed directories (data/samples, data/uploads) or traversal detected.",
        )

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"PCAP file not found: {pcap_path}")

    return target


class InvestigateRequest(BaseModel):
    flow_id: str = Field(..., min_length=1, max_length=128)
    use_llm: bool = True


class TrainRequest(BaseModel):
    capture_id: str | None = None
    mode: str | None = None


class DetectRequest(BaseModel):
    capture_id: str | None = None
    retrain: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/summary")
def summary() -> dict[str, Any]:
    return get_repo().summary_counts()


@app.get("/captures")
def list_captures() -> list[dict[str, Any]]:
    return get_repo().list_captures()


@app.post("/captures/upload")
async def upload_capture(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith((".pcap", ".pcapng")):
        raise HTTPException(status_code=400, detail="Only .pcap / .pcapng files are accepted")

    settings = load_settings()
    upload_dir = (PROJECT_ROOT / "data" / "uploads").resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    clean_name = Path(file.filename).name
    time_prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_name = f"{time_prefix}_{uuid.uuid4().hex[:8]}_{clean_name}"
    dest = upload_dir / unique_name

    total_size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds maximum size limit (50MB)")
            out.write(chunk)

    cid, df = extract_from_pcap(dest, out_csv=None, persist_db=True, settings=settings)
    return {"capture_id": cid, "flow_count": len(df), "path": str(dest), "filename": unique_name}


@app.post("/captures/extract")
def extract_local(pcap_path: str = Query(..., description="Path to offline PCAP")) -> dict[str, Any]:
    settings = load_settings()
    path = _validate_pcap_path(pcap_path, settings=settings)
    cid, df = extract_from_pcap(path, persist_db=True, settings=settings)
    return {"capture_id": cid, "flow_count": len(df)}


@app.post("/captures/ingest-unsw")
def ingest_unsw(samples: int = Query(1500, ge=100, le=5000)) -> dict[str, Any]:
    """Download and ingest benchmark network threat flows from UNSW-NB15."""
    from ettd.dataset_ingest import download_and_ingest_unsw
    settings = load_settings()
    return download_and_ingest_unsw(num_samples=samples, settings=settings)


@app.get("/flows")
def list_flows(
    capture_id: str | None = None,
    label: str | None = None,
    dst_ip: str | None = None,
    dst_port: int | None = None,
    min_score: float | None = None,
    search: str | None = None,
    sort_by: str = Query("flow_id", pattern="^(flow_id|score|start_time|dst_port)$"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$"),
    limit: int | None = Query(None, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    """Get flows joined with features and predictions in a single query with filters and pagination."""
    repo = get_repo()
    return repo.list_flows_enriched(
        capture_id=capture_id,
        label=label,
        dst_ip=dst_ip,
        dst_port=dst_port,
        min_score=min_score,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
    )


@app.get("/flows/{flow_id}")
def get_flow(flow_id: str) -> dict[str, Any]:
    repo = get_repo()
    flow = repo.get_flow(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="flow not found")
    return {
        **flow,
        "features": repo.get_flow_features(flow_id),
        "prediction": repo.get_prediction(flow_id),
        "investigations": repo.list_investigations(flow_id),
    }


@app.post("/train")
def run_train(body: TrainRequest | None = None) -> dict[str, Any]:
    settings = load_settings()
    if body and body.mode:
        settings.setdefault("ml", {})["mode"] = body.mode
    repo = get_repo()
    capture_id = body.capture_id if body else None
    df = repo.get_features_dataframe(capture_id)
    if df.empty:
        raise HTTPException(status_code=400, detail="No features available; extract a PCAP first")
    try:
        meta = train_model(df, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "trained",
        "version": meta.get("version"),
        "mode": meta.get("mode"),
        "algorithm": meta.get("algorithm"),
        "n_samples": meta.get("n_samples"),
        "score_threshold": meta.get("score_threshold", meta.get("threshold")),
        "meta": meta,
    }


@app.post("/detect")
def run_detect(body: DetectRequest | None = None) -> dict[str, Any]:
    settings = load_settings()
    repo = get_repo()
    capture_id = body.capture_id if body else None
    df = repo.get_features_dataframe(capture_id)
    if df.empty:
        raise HTTPException(status_code=400, detail="No features available; extract a PCAP first")
    try:
        preds = detect_suspicious(df, settings, retrain=False)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Please train a model first using POST /train.",
        ) from exc

    # Group by capture from features
    if capture_id:
        repo.save_predictions(capture_id, preds)
        capture_ids = [capture_id]
    else:
        flow_cap = {f["flow_id"]: f["capture_id"] for f in repo.list_flows()}
        by_cap: dict[str, list] = {}
        for p in preds:
            cid = flow_cap.get(p["flow_id"])
            if cid:
                by_cap.setdefault(cid, []).append(p)
        for cid, items in by_cap.items():
            repo.save_predictions(cid, items)
        capture_ids = list(by_cap.keys())
    return {
        "predictions": preds,
        "capture_ids": capture_ids,
        "suspicious": sum(1 for p in preds if p["label"] == "suspicious"),
        "benign": sum(1 for p in preds if p["label"] == "benign"),
        "mode": preds[0]["model_mode"] if preds else None,
    }


@app.get("/flows/{flow_id}/explain")
def explain_flow(flow_id: str) -> dict[str, Any]:
    """
    Explain model prediction for a single flow using plain numbers.

    Isolation Forest: top 5 features by absolute z-score against training scaler.
    Random Forest: top 5 features by importance * deviation.
    """
    repo = get_repo()
    features = repo.get_flow_features(flow_id)
    if not features:
        raise HTTPException(status_code=404, detail="Flow features not found")

    settings = load_settings()
    model_file, meta_path = _resolve_model_and_meta(settings)
    if not model_file.exists():
        raise HTTPException(status_code=400, detail="No trained model found to explain flow")

    import joblib

    artifact = joblib.load(model_file)
    model = artifact["model"]
    scaler = artifact["scaler"]
    cols = artifact.get("columns", [])
    mode = artifact.get("mode")
    if not mode and meta_path.exists():
        mode = json.loads(meta_path.read_text(encoding="utf-8")).get("mode", "isolation_forest")

    vals = []
    for c in cols:
        v = features.get(c, 0.0)
        try:
            val_float = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0
        except (ValueError, TypeError):
            val_float = 0.0
        vals.append(val_float)

    vals_arr = np.array(vals, dtype=float)
    means = getattr(scaler, "mean_", np.zeros_like(vals_arr))
    scales = getattr(scaler, "scale_", np.ones_like(vals_arr))
    safe_scales = np.where(scales > 1e-9, scales, 1.0)
    z_scores = (vals_arr - means) / safe_scales
    abs_z = np.abs(z_scores)

    explained_features = []
    if mode == "isolation_forest" or model.__class__.__name__ == "IsolationForest":
        top_indices = np.argsort(abs_z)[::-1][:5]
        for idx in top_indices:
            explained_features.append(
                {
                    "feature": cols[idx],
                    "value": round(float(vals_arr[idx]), 4),
                    "mean": round(float(means[idx]), 4),
                    "z_score": round(float(z_scores[idx]), 4),
                    "score": round(float(abs_z[idx]), 4),
                }
            )
    elif mode == "autoencoder" or model.__class__.__name__ == "MLPRegressor":
        xs_val = scaler.transform(vals_arr.reshape(1, -1))
        recon = model.predict(xs_val)
        recon_diff = np.square(xs_val[0] - recon[0])
        top_indices = np.argsort(recon_diff)[::-1][:5]
        for idx in top_indices:
            explained_features.append(
                {
                    "feature": cols[idx],
                    "value": round(float(vals_arr[idx]), 4),
                    "mean": round(float(means[idx]), 4),
                    "z_score": round(float(recon_diff[idx]), 4),
                    "score": round(float(recon_diff[idx]), 4),
                }
            )
    else:
        importances = getattr(model, "feature_importances_", np.ones_like(vals_arr))
        scores = importances * abs_z
        top_indices = np.argsort(scores)[::-1][:5]
        for idx in top_indices:
            explained_features.append(
                {
                    "feature": cols[idx],
                    "value": round(float(vals_arr[idx]), 4),
                    "importance": round(float(importances[idx]), 4),
                    "z_score": round(float(z_scores[idx]), 4),
                    "score": round(float(scores[idx]), 4),
                }
            )

    return {
        "flow_id": flow_id,
        "mode": mode,
        "top_features": explained_features,
    }


@app.get("/stats/score-distribution")
def get_score_distribution() -> dict[str, Any]:
    settings = load_settings()
    repo = get_repo()
    dist = repo.get_score_distribution()
    _, meta_path = _resolve_model_and_meta(settings)
    threshold = 0.55
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            threshold = float(m.get("score_threshold", m.get("threshold", 0.55)))
        except Exception:
            pass
    dist["threshold"] = threshold
    return dist


@app.get("/stats/timeline")
def get_timeline(bucket_seconds: float = Query(60.0, ge=0.001)) -> list[dict[str, Any]]:
    return get_repo().get_timeline(bucket_seconds=bucket_seconds)


@app.get("/stats/top-talkers")
def get_top_talkers(limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    return get_repo().get_top_talkers(limit=limit)


@app.post("/investigate")
def run_investigate(body: InvestigateRequest) -> dict[str, Any]:
    repo = get_repo()
    try:
        result = investigate_flow(body.flow_id, repo, use_llm=body.use_llm)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


@app.get("/investigations")
def list_investigations(flow_id: str | None = None) -> list[dict[str, Any]]:
    return get_repo().list_investigations(flow_id)


@app.get("/investigations/{investigation_id}")
def get_investigation(investigation_id: str) -> dict[str, Any]:
    inv = get_repo().get_investigation(investigation_id)
    if not inv:
        raise HTTPException(status_code=404, detail="investigation not found")
    return inv


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ettd.api.main:app", host="127.0.0.1", port=8000, reload=True)
