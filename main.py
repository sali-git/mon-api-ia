"""
API FastAPI — prédiction de priorité (High / Medium / Low).

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Colonnes attendues (même ordre que surgiflow_training_dataset.csv, sans patient_id ni cible)
FEATURE_NAMES: list[str] = [
    "age",
    "is_child",
    "sex",
    "bmi",
    "blood_type",
    "hypertension",
    "diabetes",
    "heart_disease",
    "copd",
    "renal_failure",
    "anticoagulant",
    "allergy_severity",
    "asa_score",
    "admission_urgency",
    "chief_complaint_category",
    "pain_score",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "temperature_c",
    "spo2",
    "gcs",
    "intervention_type",
    "estimated_duration_min",
    "anesthesia_risk",
    "waiting_time_hours",
]


class PredictRequest(BaseModel):
    """Une ligne de features = une prédiction. Types alignés sur le CSV d'entraînement."""

    age: int = Field(..., ge=0, le=120)
    is_child: int = Field(..., ge=0, le=1)
    sex: str
    bmi: float
    blood_type: str
    hypertension: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    copd: int = Field(..., ge=0, le=1)
    renal_failure: int = Field(..., ge=0, le=1)
    anticoagulant: int = Field(..., ge=0, le=1)
    allergy_severity: str
    asa_score: int = Field(..., ge=1, le=5)
    admission_urgency: str
    chief_complaint_category: str
    pain_score: int = Field(..., ge=0, le=10)
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    temperature_c: float
    spo2: int = Field(..., ge=70, le=100)
    gcs: int = Field(..., ge=3, le=15)
    intervention_type: str
    estimated_duration_min: int = Field(..., ge=15, le=600)
    anesthesia_risk: str
    waiting_time_hours: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    priority_level: str
    priority_score: float
    model_name: str | None = None


def _default_model_paths() -> list[Path]:
    base = Path(__file__).resolve().parent
    return [
        base / "priority_model_professional.joblib",
        base / "artifacts" / "priority_model_professional.joblib",
        base.parent / "artifacts" / "priority_model_professional.joblib",
        base.parent / "artifacts" / "pipeline_priority.joblib",
    ]


def _load_bundle() -> dict[str, Any]:
    env_path = os.getenv("MODEL_PATH", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(_default_model_paths())
    for p in candidates:
        if p.is_file():
            return joblib.load(p)
    raise FileNotFoundError(
        "Aucun fichier .joblib trouvé. Placez priority_model_professional.joblib "
        "à la racine du projet (à côté de main.py), ou dans ./artifacts/, "
        "ou définissez la variable d'environnement MODEL_PATH."
    )


def _urgency_score(proba: np.ndarray, classes: np.ndarray) -> float:
    weights = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
    wvec = np.array([weights.get(str(c), 0.5) for c in classes])
    return float((proba * wvec).sum() * 100.0)


_bundle: dict[str, Any] | None = None


def get_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        _bundle = _load_bundle()
    return _bundle


app = FastAPI(title="SurgiFlow — priorité IA", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_bundle()
        return {"status": "ok", "model": "loaded"}
    except FileNotFoundError as e:
        return {"status": "error", "detail": str(e)}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    bundle = get_bundle()
    row = body.model_dump()
    df = pd.DataFrame([row])

    if "pipeline" in bundle:
        pipe = bundle["pipeline"]
        fc = bundle.get("feature_columns")
        if fc:
            df = pd.DataFrame([{c: row[c] for c in fc}])
        pred = pipe.predict(df)[0]
        proba = pipe.predict_proba(df)[0]
        clf = pipe.named_steps["clf"]
        classes = clf.classes_
        name = bundle.get("model_name")
        return PredictResponse(
            priority_level=str(pred),
            priority_score=_urgency_score(proba, classes),
            model_name=name,
        )

    if "preprocessor" in bundle and "model" in bundle:
        prep = bundle["preprocessor"]
        model = bundle["model"]
        le: Any = bundle["label_encoder"]
        feature_cols = list(bundle.get("feature_columns", FEATURE_NAMES))
        missing = [c for c in feature_cols if c not in row]
        if missing:
            raise HTTPException(status_code=400, detail=f"Champs manquants: {missing}")
        df = pd.DataFrame([{c: row[c] for c in feature_cols}])
        Xt = prep.transform(df)
        pred_idx = model.predict(Xt)[0]
        proba = model.predict_proba(Xt)[0]
        classes = le.classes_
        label = le.inverse_transform(np.array([pred_idx]))[0]
        return PredictResponse(
            priority_level=str(label),
            priority_score=_urgency_score(proba, classes),
            model_name=bundle.get("model_name"),
        )

    raise HTTPException(
        status_code=500,
        detail="Format joblib inconnu : attendu clés 'pipeline' ou 'preprocessor'+'model'.",
    )


@app.get("/features")
def features() -> dict[str, list[str]]:
    """Liste des champs attendus par POST /predict."""
    try:
        b = get_bundle()
        cols = b.get("feature_columns")
        if cols:
            return {"feature_columns": list(cols)}
    except FileNotFoundError:
        pass
    return {"feature_columns": FEATURE_NAMES}
