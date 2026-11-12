"""FastAPI service for the Verity fake-news screening application."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .predictor import (
    ModelNotTrainedError,
    get_model_metrics,
    model_status,
    predict_news,
)

app = FastAPI(
    title="Verity — Fake News Prediction API",
    description="A TF-IDF and Logistic Regression screening model for news text.",
    version="1.2.0",
)

origins = [
    value.strip()
    for value in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=500, description="News headline")
    text: str = Field(default="", max_length=100_000, description="Optional full article body")

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Headline cannot be blank.")
        return value.strip()

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "Verity Fake News Prediction API", "docs": "/docs"}


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_ready": model_status()}


@app.get("/api/metrics")
def metrics() -> dict[str, object]:
    try:
        return get_model_metrics()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    try:
        return predict_news(title=request.title, text=request.text)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Avoid returning stack traces or filesystem details to API clients.
        raise HTTPException(status_code=500, detail="Prediction failed. Check the backend logs.") from exc
