import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).parent / "model")))
APP_VERSION = os.getenv("APP_VERSION", "dev")
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
FEATURES = ["age", "income", "loan_amount", "credit_score",
            "employment_years", "num_open_accounts", "home_ownership"]

PREDICTIONS = Counter("loan_predictions", "Predictions served", ["outcome"])
LATENCY = Histogram("loan_prediction_latency_seconds", "Model inference latency (s)")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_DIR / "model.joblib")
    state["meta"] = json.loads((MODEL_DIR / "metadata.json").read_text())
    yield
    state.clear()


app = FastAPI(title="Loan Default API", version=APP_VERSION, lifespan=lifespan)


class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    employment_years: float = Field(..., ge=0, le=60)
    num_open_accounts: int = Field(..., ge=0, le=50)
    home_ownership: Literal["RENT", "OWN", "MORTGAGE"]


class PredictionRequest(BaseModel):
    applications: list[LoanApplication] = Field(..., min_length=1, max_length=100)


class Prediction(BaseModel):
    default_probability: float
    decision: Literal["approve", "reject"]


class PredictionResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok",
            "model_version": state["meta"]["model_version"],
            "app_version": APP_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    df = pd.DataFrame([a.model_dump() for a in req.applications], columns=FEATURES)
    start = time.perf_counter()
    proba = model.predict_proba(df)[:, 1]
    LATENCY.observe(time.perf_counter() - start)

    preds = []
    for p in proba:
        is_default = float(p) >= THRESHOLD
        PREDICTIONS.labels(outcome="default" if is_default else "repay").inc()
        preds.append(Prediction(default_probability=round(float(p), 4),
                                decision="reject" if is_default else "approve"))
    return PredictionResponse(model_version=state["meta"]["model_version"],
                              threshold=THRESHOLD, predictions=preds)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
