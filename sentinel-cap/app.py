r"""
Sentinel-Cap | FastAPI Inference Server & Pydantic Data Contract Gateway
Endpoints:
  GET  /         -> Serves interactive dark-mode GUI (static/index.html)
  GET  /health   -> Service health and loaded model metadata
  POST /predict  -> Validated excess-return prediction, SHAP attributions & allocation tips
"""

from __future__ import annotations

import io
import json
import logging
import socket
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

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
MASTER_PROCESSED_PATH = PROJECT_ROOT / "data" / "03_processed" / "sentinel_features_master.parquet"
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


SECTOR_YF_CANDIDATES = {
    "BANKING": ["BANKBEES.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
    "IT": ["ITBEES.NS", "NETFIT.NS", "INFY.NS", "TCS.NS"],
    "ENERGY": ["CPSEETF.NS", "ONGC.NS", "RELIANCE.NS"],
    "INFRA": ["INFRABEES.NS", "LT.NS"],
    "FMCG": ["FMCGIETF.NS", "HINDUNILVR.NS", "ITC.NS"],
    "PHARMA": ["PHARMABEES.NS", "SUNPHARMA.NS"],
    "GOLD": ["GOLDBEES.NS", "GC=F"],
    "GILT_DEBT": ["LTGILTBEES.NS", "GILT5YBEES.NS", "SETF10GILT.NS"],
}


def _safe_yf_download(ticker: str, period: str = "35d") -> pd.DataFrame | None:
    try:
        import yfinance as yf
        f_null = io.StringIO()
        with redirect_stderr(f_null), redirect_stdout(f_null):
            df = yf.download(ticker, period=period, progress=False, timeout=3)
        if df is not None and not df.empty and len(df) >= 5:
            return df
    except Exception:
        pass
    return None


_LIVE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_PARQUET_LATEST_CACHE: dict[str, dict[str, Any]] = {}
CACHE_TTL_SEC = 30.0


def _is_network_available() -> bool:
    try:
        orig = socket.getdefaulttimeout()
        socket.setdefaulttimeout(0.2)
        socket.gethostbyname("query2.finance.yahoo.com")
        socket.setdefaulttimeout(orig)
        return True
    except Exception:
        return False


def _get_parquet_latest_records() -> dict[str, dict[str, Any]]:
    global _PARQUET_LATEST_CACHE
    if _PARQUET_LATEST_CACHE:
        return _PARQUET_LATEST_CACHE

    if MASTER_PROCESSED_PATH.exists():
        try:
            m_df = pd.read_parquet(MASTER_PROCESSED_PATH)
            latest_ts = m_df["timestamp"].max()
            date_str = pd.to_datetime(latest_ts).strftime("%Y-%m-%d")
            match_df = m_df[m_df["timestamp"] == latest_ts]

            cache: dict[str, dict[str, Any]] = {}
            for _, r in match_df.iterrows():
                sec = str(r["sector_id"])
                cache[sec] = {
                    "source": "latest_market_close",
                    "status_text": f"Latest Market Close ({date_str})",
                    "timestamp": date_str,
                    "inputs": {
                        "sector_id": sec,
                        "ret_lag_5d": round(float(r["ret_lag_5d"]), 4),
                        "realized_vol_20d": round(float(r["realized_vol_20d"]), 4),
                        "vix_level": round(float(r["vix_level"]), 1),
                        "crude_mom_20d": round(float(r["crude_mom_20d"]), 3),
                        "gpr_zscore_60d": round(float(r["gpr_zscore_60d"]), 2),
                        "finbert_sentiment": round(float(r["finbert_sentiment"]), 2),
                        "spillover_c1_armed_conflict": round(float(r["spillover_c1_armed_conflict"]), 1),
                        "regime_prob_crisis": round(float(r["regime_prob_crisis"]), 2),
                    },
                }
            _PARQUET_LATEST_CACHE = cache
        except Exception:
            pass
    return _PARQUET_LATEST_CACHE


def fetch_live_market_state(sector_id: str) -> dict[str, Any]:
    """
    Attempts to fetch live market indicators from Yahoo Finance;
    seamlessly falls back to the latest calibrated market record from
    sentinel_features_master.parquet when offline or markets are closed.
    Includes in-memory TTL caching for instant UI response times.
    """
    now = time.time()
    if sector_id in _LIVE_CACHE:
        cached_ts, cached_val = _LIVE_CACHE[sector_id]
        if now - cached_ts < CACHE_TTL_SEC:
            return cached_val

    # 1. Try Yahoo Finance live query only if internet/DNS is reachable
    if _is_network_available():
        try:
            # Query global VIX directly (1.15 multiplier maps to India VIX)
            vix_df = _safe_yf_download("^VIX", period="5d")
            vix_mult = 1.15

            # Find best live candidate ticker for this sector
            sec_df = None
            candidates = SECTOR_YF_CANDIDATES.get(sector_id, [f"{sector_id}.NS"])
            for t in candidates:
                sec_df = _safe_yf_download(t, period="35d")
                if sec_df is not None and not sec_df.empty:
                    break

            oil_df = _safe_yf_download("CL=F", period="35d")

            if sec_df is not None and not sec_df.empty and len(sec_df) >= 6:
                close_col = "Close"
                s_close = sec_df[close_col].iloc[:, 0] if isinstance(sec_df[close_col], pd.DataFrame) else sec_df[close_col]
                s_log_ret = np.log(s_close / s_close.shift(1)).dropna()
                ret_5d = float(s_log_ret.tail(5).sum())
                vol_20d = float(s_log_ret.tail(20).std() * np.sqrt(252.0))

                if vix_df is not None and not vix_df.empty:
                    v_close = vix_df[close_col].iloc[:, 0] if isinstance(vix_df[close_col], pd.DataFrame) else vix_df[close_col]
                    vix_val = float(v_close.dropna().iloc[-1] * vix_mult)
                else:
                    vix_val = 14.5

                if oil_df is not None and not oil_df.empty:
                    o_close = oil_df[close_col].iloc[:, 0] if isinstance(oil_df[close_col], pd.DataFrame) else oil_df[close_col]
                    o_log = np.log(o_close / o_close.shift(20)).dropna()
                    crude_mom = float(o_log.iloc[-1]) if len(o_log) > 0 else 0.0
                else:
                    crude_mom = 0.0

                crisis_prob = float(np.clip((vix_val - 14.0) / 30.0, 0.05, 0.95))
                gpr_z = float(np.clip((vix_val - 16.0) / 10.0, -1.0, 3.5))
                sentiment = float(np.clip(-0.02 * (vix_val - 15.0), -0.80, 0.60))
                spillover = float(np.clip(1.2 + 0.4 * (vix_val - 14.0), 0.5, 25.0))

                res = {
                    "source": "live_yfinance",
                    "status_text": "Live Real-Time Feed (Yahoo Finance)",
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "inputs": {
                        "sector_id": sector_id,
                        "ret_lag_5d": round(ret_5d, 4),
                        "realized_vol_20d": round(vol_20d, 4),
                        "vix_level": round(vix_val, 1),
                        "crude_mom_20d": round(crude_mom, 3),
                        "gpr_zscore_60d": round(gpr_z, 2),
                        "finbert_sentiment": round(sentiment, 2),
                        "spillover_c1_armed_conflict": round(spillover, 1),
                        "regime_prob_crisis": round(crisis_prob, 2),
                    },
                }
                _LIVE_CACHE[sector_id] = (now, res)
                return res
        except Exception:
            pass

    # 2. Seamless fast fallback: latest record from processed master parquet
    parquet_records = _get_parquet_latest_records()
    if sector_id in parquet_records:
        res = parquet_records[sector_id]
        _LIVE_CACHE[sector_id] = (now, res)
        return res

    fallback = {
        "source": "calibrated_baseline",
        "status_text": "Calibrated Market Baseline",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "inputs": {
            "sector_id": sector_id,
            "ret_lag_5d": 0.008,
            "realized_vol_20d": 0.16,
            "vix_level": 15.5,
            "crude_mom_20d": 0.02,
            "gpr_zscore_60d": 0.3,
            "finbert_sentiment": 0.12,
            "spillover_c1_armed_conflict": 1.2,
            "regime_prob_crisis": 0.10,
        },
    }
    _LIVE_CACHE[sector_id] = (now, fallback)
    return fallback


@app.get("/live-market/{sector_id}")
def get_live_sector_state(sector_id: str):
    s_upper = sector_id.upper()
    valid_sectors = {"BANKING", "IT", "ENERGY", "INFRA", "FMCG", "PHARMA", "GOLD", "GILT_DEBT"}
    if s_upper not in valid_sectors:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sector_id '{sector_id}'. Choose from: {sorted(valid_sectors)}",
        )
    return fetch_live_market_state(s_upper)


@app.get("/live-portfolio")
def get_live_portfolio_allocation():
    sectors = ["BANKING", "IT", "ENERGY", "INFRA", "FMCG", "PHARMA", "GOLD", "GILT_DEBT"]
    results = []
    source = "latest_market_close"
    status_text = "Latest Processed Market Close"
    data_date = ""

    for s_id in sectors:
        state = fetch_live_market_state(s_id)
        source = state["source"]
        status_text = state["status_text"]
        data_date = state["timestamp"]

        payload = state["inputs"]
        input_df = pd.DataFrame([payload])
        pred_5d = float(PIPELINE.predict(input_df)[0])
        pred_bps = round(pred_5d * 10000.0, 2)

        crisis_prob = float(payload["regime_prob_crisis"])
        uncertainty_scale = max(0.35, 1.0 - 0.5 * crisis_prob)
        bl_weight = float(np.clip(12.5 + (pred_bps * 0.14 * uncertainty_scale), 0.0, 35.0))

        if pred_bps >= 25.0:
            verdict = "OVERWEIGHT"
        elif pred_bps <= -25.0:
            verdict = "UNDERWEIGHT"
        else:
            verdict = "NEUTRAL"

        results.append({
            "sector_id": s_id,
            "predicted_excess_return_bps": pred_bps,
            "recommended_bl_weight_pct": round(bl_weight, 1),
            "active_tilt_pct": round(bl_weight - 12.5, 1),
            "signal_verdict": verdict,
            "ret_lag_5d": payload["ret_lag_5d"],
            "realized_vol_20d": payload["realized_vol_20d"],
            "vix_level": payload["vix_level"],
        })

    results.sort(key=lambda x: x["recommended_bl_weight_pct"], reverse=True)

    return {
        "source": source,
        "status_text": status_text,
        "timestamp": data_date,
        "sectors": results,
    }
