r"""
Sentinel-Cap | FastAPI Inference Server & Pydantic Data Contract Gateway
Endpoints:
  GET  /         -> Serves interactive dark-mode GUI (static/index.html)
  GET  /health   -> Service health and loaded model metadata
  POST /predict  -> Validated excess-return prediction, SHAP attributions & allocation tips
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.joblib"
STATIC_DIR = PROJECT_ROOT / "static"
LOG_DIR = PROJECT_ROOT / "logs"
PREDICTION_LOG_FILE = LOG_DIR / "prediction_inputs.jsonl"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model artifact missing at {MODEL_PATH}. Run `python train.py` first.")

MODEL_BUNDLE = joblib.load(MODEL_PATH)
PIPELINE = MODEL_BUNDLE["pipeline"]
METADATA = MODEL_BUNDLE["metadata"]

PREPROCESSOR = PIPELINE.named_steps["preprocessor"]
REGRESSOR = PIPELINE.named_steps["regressor"]
try:
    SHAP_EXPLAINER = shap.TreeExplainer(REGRESSOR)
except Exception:
    SHAP_EXPLAINER = None

app = FastAPI(
    title="Sentinel-Cap Geopolitical Sector Allocation API",
    description="Hybrid Econometric-ML Sector Excess Return Forecaster & Black-Litterman Signal Engine",
    version="1.0.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SectorLiteral = Literal[
    "BANKING", "IT", "ENERGY", "INFRA", "FMCG", "PHARMA", "GOLD", "GILT_DEBT"
]


class PredictionRequest(BaseModel):
    sector_id: SectorLiteral = Field(..., description="Target NSE sector or safe-haven proxy")
    ret_lag_5d: float = Field(..., ge=-0.50, le=0.50, description="Trailing 5-day sector log return")
    realized_vol_20d: float = Field(..., ge=0.01, le=1.50, description="Annualized 20-day realized volatility")
    vix_level: float = Field(..., ge=8.0, le=90.0, description="India VIX implied volatility level")
    crude_mom_20d: float = Field(..., ge=-1.00, le=1.00, description="20-day Brent crude oil momentum")
    gpr_zscore_60d: float = Field(..., ge=-4.0, le=10.0, description="Caldara GPR 60-day rolling z-score")
    finbert_sentiment: float = Field(..., ge=-1.0, le=1.0, description="Net FinBERT headline sentiment [-1, 1]")
    spillover_c1_armed_conflict: float = Field(
        ..., ge=0.0, le=50.0, description="Half-life decayed armed-conflict spillover intensity"
    )
    regime_prob_crisis: float = Field(
        ..., ge=0.0, le=1.0, description="Causal HMM posterior probability of Crisis regime"
    )


class FeatureAttribution(BaseModel):
    feature: str
    impact_bps: float


class PredictionResponse(BaseModel):
    sector_id: str
    predicted_excess_return_5d: float
    predicted_excess_return_bps: float
    signal_verdict: str
    recommended_bl_weight_pct: float
    regime_state: str
    top_shap_drivers: list[FeatureAttribution]
    allocation_tips: list[str]
    model_version: str


def generate_rule_based_tips(req: PredictionRequest, pred_bps: float) -> list[str]:
    tips: list[str] = []

    if req.regime_prob_crisis >= 0.60 or req.vix_level >= 25.0:
        tips.append(
            f"High-Volatility Regime Detected (Crisis Prob {req.regime_prob_crisis:.0%}, VIX {req.vix_level:.1f}): "
            "Increase Black-Litterman risk aversion lambda(Z_t) and cap cyclical equity exposure at 15% per sector."
        )
    else:
        tips.append(
            f"Calm/Normal Volatility Regime (Crisis Prob {req.regime_prob_crisis:.0%}, VIX {req.vix_level:.1f}): "
            "Standard Black-Litterman view confidence applies; maintain diversified cross-sector exposure."
        )

    if req.spillover_c1_armed_conflict >= 8.0 or req.gpr_zscore_60d >= 1.5:
        if req.sector_id in {"GOLD", "GILT_DEBT", "ENERGY", "PHARMA"}:
            tips.append(
                f"Geopolitical Tailwind for {req.sector_id}: Armed conflict spillover ({req.spillover_c1_armed_conflict:.1f}) "
                "historically favors safe havens and energy hedges—tilt overweight via GOLDBEES/CPSEETF."
            )
        else:
            tips.append(
                f"Geopolitical Headwind for {req.sector_id}: Elevated conflict spillover ({req.spillover_c1_armed_conflict:.1f}) "
                "pressures rate- and import-sensitive cyclicals—rotate excess weight into GOLDBEES or GILTBEES."
            )
    else:
        tips.append(
            f"Contained Geopolitical Spillover ({req.spillover_c1_armed_conflict:.1f}): "
            "Macro earnings momentum and FinBERT polarity dominate short-horizon sector dispersion."
        )

    if pred_bps >= 35.0:
        tips.append(
            f"Strong Positive Alpha View (+{pred_bps:.1f} bps over 5d): Execute gradual TWAP rebalance "
            "to keep turnover penalty below the 15 bps transaction-cost threshold."
        )
    elif pred_bps <= -35.0:
        tips.append(
            f"Negative Alpha View ({pred_bps:.1f} bps over 5d): Underweight {req.sector_id} relative to NIFTY 50 "
            "benchmark and park defensive liquidity in LIQUIDBEES/GILTBEES."
        )
    else:
        tips.append(
            f"Neutral Dispersion ({pred_bps:+.1f} bps over 5d): Expected alpha lies within the 15 bps round-trip "
            "transaction-cost band—hold existing Black-Litterman weights to avoid unnecessary turnover."
        )

    return tips[:3]


def compute_local_attributions(input_df: pd.DataFrame) -> list[FeatureAttribution]:
    X_enc = PREPROCESSOR.transform(input_df)
    encoded_names = METADATA.get("encoded_feature_names", [])

    if SHAP_EXPLAINER is not None:
        shap_vals = SHAP_EXPLAINER.shap_values(X_enc)
        vals = np.asarray(shap_vals).reshape(-1)
    else:
        coefs = getattr(REGRESSOR, "coef_", np.zeros(X_enc.shape[1]))
        vals = np.asarray(coefs).reshape(-1) * np.asarray(X_enc).reshape(-1)

    sector_impact = 0.0
    drivers: list[tuple[str, float]] = []
    for name, val in zip(encoded_names, vals):
        if name.startswith("sector_id_"):
            sector_impact += float(val) * 10000.0
        else:
            drivers.append((name, float(val) * 10000.0))

    drivers.append(("sector_id", sector_impact))
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)
    return [FeatureAttribution(feature=k, impact_bps=round(v, 2)) for k, v in drivers[:5]]


def log_prediction_input(payload: dict, pred_5d: float) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": payload,
        "predicted_excess_return_5d": pred_5d,
    }
    with open(PREDICTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/", include_in_schema=False)
def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI file static/index.html not found.")
    return FileResponse(str(index_file))


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "sentinel-cap-api",
        "model_type": METADATA.get("model_type"),
        "validation_rmse": round(METADATA.get("rmse", 0.0), 5),
        "validation_rank_ic": round(METADATA.get("rank_ic", 0.0), 4),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sector_allocation(req: PredictionRequest):
    payload = req.model_dump()
    input_df = pd.DataFrame([payload])

    pred_5d = float(PIPELINE.predict(input_df)[0])
    pred_bps = round(pred_5d * 10000.0, 2)

    base_weight = 12.5
    uncertainty_scale = max(0.35, 1.0 - 0.5 * req.regime_prob_crisis)
    bl_weight = float(np.clip(base_weight + (pred_bps * 0.14 * uncertainty_scale), 0.0, 35.0))

    if pred_bps >= 25.0:
        verdict = "OVERWEIGHT (BUY TILT)"
    elif pred_bps <= -25.0:
        verdict = "UNDERWEIGHT (REDUCE)"
    else:
        verdict = "NEUTRAL (BENCHMARK)"

    if req.regime_prob_crisis >= 0.60:
        regime_state = "CRISIS REGIME"
    elif req.regime_prob_crisis >= 0.30:
        regime_state = "STRESSED REGIME"
    else:
        regime_state = "CALM REGIME"

    drivers = compute_local_attributions(input_df)
    tips = generate_rule_based_tips(req, pred_bps)
    log_prediction_input(payload, pred_5d)

    return PredictionResponse(
        sector_id=req.sector_id,
        predicted_excess_return_5d=round(pred_5d, 6),
        predicted_excess_return_bps=pred_bps,
        signal_verdict=verdict,
        recommended_bl_weight_pct=round(bl_weight, 1),
        regime_state=regime_state,
        top_shap_drivers=drivers,
        allocation_tips=tips,
        model_version=f"sentinel-cap-{METADATA.get('model_type', 'xgb')}-v1",
    )
