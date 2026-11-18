from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "sentinel-cap-api"


def test_ui_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "SENTINEL-" in response.text
    assert "Black-Litterman" in response.text


def test_domain_logic_crisis_gold_beats_banking():
    base_shock = {
        "ret_lag_5d": 0.00,
        "realized_vol_20d": 0.28,
        "vix_level": 32.0,
        "crude_mom_20d": 0.18,
        "gpr_zscore_60d": 3.0,
        "finbert_sentiment": -0.55,
        "spillover_c1_armed_conflict": 16.0,
        "regime_prob_crisis": 0.90,
    }
    res_gold = client.post("/predict", json={"sector_id": "GOLD", **base_shock})
    res_bank = client.post("/predict", json={"sector_id": "BANKING", **base_shock})
    assert res_gold.status_code == 200
    assert res_bank.status_code == 200
    assert res_gold.json()["predicted_excess_return_5d"] > res_bank.json()["predicted_excess_return_5d"]


def test_boundary_out_of_range_vix_returns_422():
    invalid_payload = {
        "sector_id": "ENERGY",
        "ret_lag_5d": 0.01,
        "realized_vol_20d": 0.22,
        "vix_level": 250.0,
        "crude_mom_20d": 0.05,
        "gpr_zscore_60d": 1.2,
        "finbert_sentiment": -0.1,
        "spillover_c1_armed_conflict": 4.0,
        "regime_prob_crisis": 0.4,
    }
    assert client.post("/predict", json=invalid_payload).status_code == 422


def test_invalid_sector_returns_422():
    invalid_payload = {
        "sector_id": "CRYPTO_MEME",
        "ret_lag_5d": 0.01,
        "realized_vol_20d": 0.22,
        "vix_level": 18.0,
        "crude_mom_20d": 0.05,
        "gpr_zscore_60d": 0.5,
        "finbert_sentiment": 0.1,
        "spillover_c1_armed_conflict": 1.0,
        "regime_prob_crisis": 0.1,
    }
    assert client.post("/predict", json=invalid_payload).status_code == 422


def test_live_market_endpoint():
    res = client.get("/live-market/GOLD")
    assert res.status_code == 200
    data = res.json()
    assert "source" in data
    assert "status_text" in data
    assert "inputs" in data
    assert data["inputs"]["sector_id"] == "GOLD"
    assert "vix_level" in data["inputs"]

    res_invalid = client.get("/live-market/INVALID_TICKER")
    assert res_invalid.status_code == 400


def test_live_portfolio_endpoint():
    res = client.get("/live-portfolio")
    assert res.status_code == 200
    data = res.json()
    assert "sectors" in data
    assert len(data["sectors"]) == 8
    sector_ids = {s["sector_id"] for s in data["sectors"]}
    expected_sectors = {"BANKING", "IT", "ENERGY", "INFRA", "FMCG", "PHARMA", "GOLD", "GILT_DEBT"}
    assert sector_ids == expected_sectors
    for s in data["sectors"]:
        assert "recommended_bl_weight_pct" in s
        assert "active_tilt_pct" in s
        assert "signal_verdict" in s
