"""ZEIS API. Models are loaded once at startup; request handlers never train anything."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas import (AnalyzeResponse, BatchRequest, BatchResponse, CategorizeRequest, CategorizeResponse,
                         ExpenseRequest, HealthResponse)
from src.inference import Predictor

log = logging.getLogger("zeis")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = Predictor()
        log.info("models loaded, version %s", app.state.predictor.version)
    except Exception:  # keep the process up so /health can report the problem
        log.exception("could not load model artifacts")
        app.state.predictor = None
    yield


app = FastAPI(
    title="ZEIS - Zubera Expense Intelligence Service",
    description=("Expense categorization and anomaly scoring trained on SYNTHETIC data. Scores are model scores, not calibrated "
                 "probabilities. Output is never proof of fraud and never authorises reimbursement."),
    version="1.0.0", lifespan=lifespan)


def get_predictor(request: Request) -> Predictor:
    p = request.app.state.predictor
    if p is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded. Run `python -m src.train` first.")
    return p


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    p = request.app.state.predictor
    if p is None:
        return JSONResponse(status_code=503, content={"status": "degraded", "models_loaded": False})
    return HealthResponse(status="ok", models_loaded=True, model_version=p.version, model_a=p.meta["model_a"], model_b=p.meta["model_b"])


@app.post("/categorize", response_model=CategorizeResponse)
def categorize(body: CategorizeRequest, request: Request):
    p = get_predictor(request)
    out = p.categorize([body.model_dump()])[0]
    return CategorizeResponse(**out, model_version=p.version)


@app.post("/analyze-expense", response_model=AnalyzeResponse)
def analyze_expense(body: ExpenseRequest, request: Request, record_history: bool = False):
    p = get_predictor(request)
    return p.analyze([body.model_dump(mode="json")], record_history=record_history)[0]


@app.post("/analyze-batch", response_model=BatchResponse)
def analyze_batch(body: BatchRequest, request: Request):
    p = get_predictor(request)
    res = p.analyze([e.model_dump(mode="json") for e in body.expenses])
    summary = {t: sum(r["risk_tier"] == t for r in res) for t in ("approve", "review", "flag")}
    return BatchResponse(count=len(res), summary=summary, results=res)
