"""FastAPI backend for ETTD."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from ettd.agent.investigator import investigate_flow
from ettd.config import PROJECT_ROOT, db_path, load_settings
from ettd.db.repository import Repository
from ettd.ml.infer import detect_suspicious
from ettd.ml.train import train_model
from ettd.pipeline_extract import extract_from_pcap

app = FastAPI(
    title="Encrypted Traffic Threat Detection API",
    description="Metadata-only encrypted traffic analysis. No decryption. No auto-blocking.",
    version="0.1.0",
)


def get_repo() -> Repository:
    return Repository(db_path(load_settings()))


class InvestigateRequest(BaseModel):
    flow_id: str = Field(..., min_length=1, max_length=128)
    use_llm: bool = True


class TrainRequest(BaseModel):
    capture_id: str | None = None


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
    upload_dir = PROJECT_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    cid, df = extract_from_pcap(dest, out_csv=None, persist_db=True, settings=settings)
    return {"capture_id": cid, "flow_count": len(df), "path": str(dest)}


@app.post("/captures/extract")
def extract_local(pcap_path: str = Query(..., description="Path to offline PCAP")) -> dict[str, Any]:
    path = Path(pcap_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise HTTPException(status_code=404, detail="PCAP not found")
    cid, df = extract_from_pcap(path, persist_db=True)
    return {"capture_id": cid, "flow_count": len(df)}


@app.get("/flows")
def list_flows(capture_id: str | None = None) -> list[dict[str, Any]]:
    repo = get_repo()
    flows = repo.list_flows(capture_id)
    enriched = []
    for flow in flows:
        feat = repo.get_flow_features(flow["flow_id"]) or {}
        pred = repo.get_prediction(flow["flow_id"])
        enriched.append({**flow, "features": feat, "prediction": pred})
    return enriched


@app.get("/flows/{flow_id}")
def get_flow(flow_id: str) -> dict[str, Any]:
    repo = get_repo()
    flows = [f for f in repo.list_flows() if f["flow_id"] == flow_id]
    if not flows:
        raise HTTPException(status_code=404, detail="flow not found")
    return {
        **flows[0],
        "features": repo.get_flow_features(flow_id),
        "prediction": repo.get_prediction(flow_id),
        "investigations": repo.list_investigations(flow_id),
    }


@app.post("/train")
def run_train(body: TrainRequest | None = None) -> dict[str, Any]:
    settings = load_settings()
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
        # Save per flow's capture
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
