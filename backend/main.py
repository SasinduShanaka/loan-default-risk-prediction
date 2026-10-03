"""FastAPI application for loan status prediction."""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.model_service import LoanPredictionService
from backend.validation import ValidationError, validate_payload


app = FastAPI(title="Loan Status Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

service = LoanPredictionService()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Loan Status Prediction API",
        "status": "running",
        "model": service.metadata["selected_model"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": service.ready,
        "model": service.metadata["selected_model"],
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return dict(service.metadata)


@app.post("/predict")
def predict(payload: Any = Body(...)) -> dict[str, Any]:
    try:
        cleaned_payload = validate_payload(payload, service.metadata)
        return service.predict(cleaned_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Prediction failed unexpectedly"
        ) from exc
