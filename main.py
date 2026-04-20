"""
API FastAPI — prédiction de priorité (High / Medium / Low).

Charge `surgiflow_priority_bundle.joblib` (notebook SurgiFlow) : clés
`model_rf`, `label_encoder`, `meta` — ou anciens formats `pipeline` / preprocessor+model.
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Colonnes attendues (ordre Angular / CSV complet ; le modèle utilise `meta.feature_columns`)
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
        base / "surgiflow_priority_bundle.joblib",
        base / "priority_model_professional.joblib",
        base / "artifacts" / "surgiflow_priority_bundle.joblib",
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
        "Aucun fichier .joblib trouvé. Placez surgiflow_priority_bundle.joblib "
        "à la racine (à côté de main.py), ou définissez MODEL_PATH."
    )


def _urgency_score(proba: np.ndarray, class_labels_for_columns: list[str]) -> float:
    """proba[i] = classe i ; libellés Low/Medium/High pour le score."""
    weights = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
    wvec = np.array([weights.get(str(c), 0.5) for c in class_labels_for_columns])
    if len(wvec) != len(proba):
        raise ValueError(
            f"proba ({len(proba)}) et libellés ({len(wvec)}) : ordre des classes incohérent."
        )
    return float((proba * wvec).sum() * 100.0)


def _hybrid_boost(row: dict[str, Any], label_pred: str, proba_high: float) -> tuple[str, float]:
    """Règles post-modèle (alignées sur le notebook) — ne remplacent pas la décision médicale."""
    spo2 = float(row.get("spo2", 99))
    adm = str(row.get("admission_urgency", "")).lower()
    pain = float(row.get("pain_score", 0))
    if spo2 < 88 and label_pred != "High":
        return "High", max(proba_high, 0.55)
    if adm == "emergency" and label_pred == "Medium":
        return "High", max(proba_high, 0.5)
    if spo2 < 92 and pain >= 8 and label_pred == "Low":
        return "Medium", proba_high
    return label_pred, proba_high


def _model_classes_to_priority_strings(model: Any, le: Any) -> list[str]:
    out: list[str] = []
    for c in np.asarray(model.classes_).ravel():
        if isinstance(c, (np.integer, int)):
            out.append(str(le.inverse_transform(np.array([int(c)]))[0]))
        else:
            out.append(str(c))
    return out


def _prediction_to_label(pred: Any, le: Any) -> str:
    if isinstance(pred, (np.integer, int, np.floating)) or isinstance(pred, float):
        v = int(pred)
        return str(le.inverse_transform(np.array([v]))[0])
    return str(pred)


def _predict_surgiflow_bundle(bundle: dict[str, Any], row: dict[str, Any]) -> PredictResponse:
    """Bundle issu du notebook : model_rf = Pipeline, label_encoder, meta.feature_columns."""
    pipe: Any = bundle["model_rf"]
    le: Any = bundle["label_encoder"]
    meta = bundle.get("meta") or {}
    fc = meta.get("feature_columns")
    if not fc:
        raise HTTPException(
            status_code=500,
            detail="Bundle incomplet : meta.feature_columns absent.",
        )
    missing = [c for c in fc if c not in row]
    if missing:
        raise HTTPException(status_code=400, detail=f"Champs manquants: {missing}")

    df = pd.DataFrame([{c: row[c] for c in fc}])
    clf = pipe.named_steps["clf"]
    pred_idx = int(pipe.predict(df)[0])
    label = str(le.inverse_transform(np.asarray([pred_idx]))[0])
    proba = pipe.predict_proba(df)[0]
    str_labels = [
        str(le.inverse_transform(np.asarray([int(ci)]))[0]) for ci in clf.classes_
    ]
    score = _urgency_score(proba, str_labels)
    try:
        proba_high = float(proba[str_labels.index("High")])
    except ValueError:
        proba_high = 0.0

    label_f, _ = _hybrid_boost(row, label, proba_high)
    if label_f != label:
        if label_f == "High":
            score = max(score, 72.0)
        elif label_f == "Medium":
            score = max(score, 48.0)

    return PredictResponse(
        priority_level=label_f,
        priority_score=round(float(score), 1),
        model_name="random_forest",
    )


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
    try:
        bundle = get_bundle()
        row = body.model_dump()

        # Notebook SurgiFlow — surgiflow_priority_bundle.joblib
        if "model_rf" in bundle and "label_encoder" in bundle:
            return _predict_surgiflow_bundle(bundle, row)

        df = pd.DataFrame([row])

        if "pipeline" in bundle:
            pipe = bundle["pipeline"]
            fc = bundle.get("feature_columns")
            if fc:
                df = pd.DataFrame([{c: row[c] for c in fc}])
            pred = pipe.predict(df)[0]
            clf = pipe.named_steps["clf"]
            str_labels = [str(c) for c in clf.classes_]
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(df)[0]
                score = _urgency_score(proba, str_labels)
            else:
                wm = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
                score = wm.get(str(pred), 0.5) * 100.0
            name = bundle.get("model_name")
            return PredictResponse(
                priority_level=str(pred),
                priority_score=score,
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
            pred_raw = model.predict(Xt)[0]
            label = _prediction_to_label(pred_raw, le)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xt)[0]
                str_labels = _model_classes_to_priority_strings(model, le)
                score = _urgency_score(proba, str_labels)
            else:
                wm = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
                score = wm.get(str(label), 0.5) * 100.0

            return PredictResponse(
                priority_level=str(label),
                priority_score=score,
                model_name=bundle.get("model_name"),
            )

        raise HTTPException(
            status_code=500,
            detail="Format joblib inconnu : attendu bundle SurgiFlow (model_rf), "
            "'pipeline', ou 'preprocessor'+'model'.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("POST /predict")
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ) from e


@app.get("/features")
def features() -> dict[str, list[str]]:
    """Liste des champs utilisés par le modèle (meta.feature_columns) ou schéma complet."""
    try:
        b = get_bundle()
        if "meta" in b and isinstance(b["meta"], dict):
            cols = b["meta"].get("feature_columns")
            if cols:
                return {"feature_columns": list(cols)}
        cols = b.get("feature_columns")
        if cols:
            return {"feature_columns": list(cols)}
    except FileNotFoundError:
        pass
    return {"feature_columns": FEATURE_NAMES}
