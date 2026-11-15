"""
Sentinel-Cap | Sprint 1 (Patched): Full 25-Year (2000-2025) Indian Sectoral & Global Macro Extraction
Splices ^BSESN log-returns into ^NSEI prior to Sept 2007 so 2000-2007 is 100% preserved.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

SECTOR_UNIVERSE = {
    "BANKING": {
        "index": "^NSEBANK",
        "etf": "BANKBEES.NS",
        "proxies": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
    },
    "IT": {
        "index": "^CNXIT",
        "etf": "SETFNIFIT.NS",
        "proxies": ["INFY.NS", "WIPRO.NS"],
    },
    "ENERGY": {
        "index": "^CNXENERGY",
        "etf": "CPSEETF.NS",
        "proxies": ["RELIANCE.NS", "ONGC.NS", "IOC.NS"],
    },
    "INFRA": {
        "index": "^CNXINFRA",
        "etf": "INFRAIETF.NS",
        "proxies": ["LT.NS", "NTPC.NS", "TATAPOWER.NS"],
    },
    "FMCG": {
        "index": "^CNXFMCG",
        "etf": "FMCGIETF.NS",
        "proxies": ["HINDUNILVR.NS", "ITC.NS"],
    },
    "PHARMA": {
        "index": "^CNXPHARMA",
        "etf": "PHARMABEES.NS",
        "proxies": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"],
    },
    "GOLD": {
        "index": "GC=F",
        "etf": "GOLDBEES.NS",
        "proxies": ["GC=F"],
    },
    "GILT_DEBT": {
        "index": "^TNX",
        "etf": "GILTBEES.NS",
        "proxies": ["^TNX"],
    },
}

BENCHMARK_TICKER = "^NSEI"
FALLBACK_BENCHMARK = "^BSESN"  # Full daily history from 1997 on Yahoo Finance

MACRO_CONTROLS = {
    "india_vix": "^INDIAVIX",
    "us_vix": "^VIX",
    "crude_oil": "CL=F",
    "usd_inr": "INR=X",
    "us_10y_yield": "^TNX",
    "sp500": "^GSPC",
}


def download_single_close(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    col = "Close" if "Close" in df.columns else df.columns[0]
    s = df[col].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s


def _log_returns(s: pd.Series, periods: int = 1) -> pd.Series:
    """Computes log-returns ln(P_t / P_{t-k}) as a typed pandas Series."""
    ratio = s / s.shift(periods)
    return pd.Series(np.log(ratio), index=s.index)


def _forward_sum(s: pd.Series, h: int) -> pd.Series:
    """Computes forward cumulative sum sum_{k=1}^h s_{t+k} as a typed pandas Series."""
    fwd = s.shift(-1).copy()
    for k in range(2, h + 1):
        fwd = fwd + s.shift(-k)
    return fwd


def build_spliced_benchmark(start: str = "1999-01-01", end: str = "2026-01-01") -> pd.Series:
    """Combines ^NSEI (2007-2026) with ^BSESN log-returns (1999-2007) for unbroken 25-year history."""
    nsei = download_single_close(BENCHMARK_TICKER, start, end).dropna()
    bsesn = download_single_close(FALLBACK_BENCHMARK, start, end).dropna()

    master_idx = nsei.index.union(bsesn.index).sort_values()
    nsei_ret = _log_returns(nsei.reindex(master_idx))
    bsesn_ret = _log_returns(bsesn.reindex(master_idx))

    combined_ret = nsei_ret.combine_first(bsesn_ret).fillna(0.0)
    reconstructed_close = pd.Series(
        1000.0 * np.exp(combined_ret.cumsum()),
        index=master_idx,
        name=BENCHMARK_TICKER,
    )
    return reconstructed_close


def build_continuous_sector_returns(
    master_dates: pd.DatetimeIndex, start: str, end: str
) -> pd.DataFrame:
    sector_log_rets = pd.DataFrame(index=master_dates)

    for sector_name, cfg in SECTOR_UNIVERSE.items():
        print(f"  -> Processing Sector: {sector_name} ({cfg['index']})...")
        idx_series = download_single_close(cfg["index"], start, end).reindex(master_dates)

        if sector_name == "GILT_DEBT":
            yield_lvl = idx_series.ffill().bfill()
            daily_yield_diff = yield_lvl.diff() / 100.0
            carry = (yield_lvl.shift(1) / 100.0) / 252.0
            bond_ret = carry - 6.5 * daily_yield_diff
            sector_log_rets[sector_name] = bond_ret.fillna(0.0).clip(-0.10, 0.10)
            continue

        idx_ret = _log_returns(idx_series)

        if idx_ret.isna().sum() > 10 and cfg["proxies"] != [cfg["index"]]:
            proxy_rets = []
            for p_tk in cfg["proxies"]:
                p_s = download_single_close(p_tk, start, end).reindex(master_dates).ffill()
                if not p_s.empty:
                    proxy_rets.append(_log_returns(p_s))
            if proxy_rets:
                basket_ret = pd.concat(proxy_rets, axis=1).mean(axis=1)
                idx_ret = idx_ret.combine_first(basket_ret)

        sector_log_rets[sector_name] = idx_ret.fillna(0.0).clip(-0.20, 0.20)

    return sector_log_rets


def build_market_and_macro_pipeline(
    start_date: str = "1999-01-01",
    end_date: str = "2026-01-01",
    raw_dir: str = "data/01_raw",
    interim_dir: str = "data/02_interim",
) -> pd.DataFrame:
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(interim_dir).mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Building Continuous 25-Year Indian Benchmark (^NSEI + ^BSESN pre-2007 splice)...")
    bench_close = build_spliced_benchmark(start_date, end_date)
    master_dates = pd.DatetimeIndex(bench_close.index)
    bench_ret = _log_returns(bench_close).fillna(0.0)

    print("[2/4] Downloading & Backfilling 8 Indian Sector / Safe-Haven Series...")
    sector_rets = build_continuous_sector_returns(master_dates, start_date, end_date)

    print("[3/4] Downloading Global & Domestic Macroeconomic Controls...")
    macro_df = pd.DataFrame(index=master_dates)
    for m_name, m_tk in MACRO_CONTROLS.items():
        s = download_single_close(m_tk, start_date, end_date).reindex(master_dates).ffill()
        macro_df[m_name] = s

    if macro_df["india_vix"].isna().any():
        vix_ratio = (macro_df["india_vix"] / macro_df["us_vix"]).median() or 1.15
        macro_df["india_vix"] = macro_df["india_vix"].combine_first(macro_df["us_vix"] * vix_ratio)
    macro_df = macro_df.ffill().bfill()

    macro_features = pd.DataFrame(index=master_dates)
    macro_features["bench_log_ret"] = bench_ret
    macro_features["bench_vol_20d"] = bench_ret.rolling(20).std() * np.sqrt(252)
    sma50 = bench_close.rolling(50).mean()
    sma200 = bench_close.rolling(200).mean()
    macro_features["bench_sma50_200_ratio"] = (sma50 / sma200) - 1.0
    macro_features["vix_level"] = macro_df["india_vix"]
    macro_features["vix_change_5d"] = macro_df["india_vix"].pct_change(5).fillna(0.0)
    macro_features["crude_ret_1d"] = _log_returns(macro_df["crude_oil"], 1).fillna(0.0)
    macro_features["crude_mom_20d"] = _log_returns(macro_df["crude_oil"], 20).fillna(0.0)
    macro_features["usdinr_ret_1d"] = _log_returns(macro_df["usd_inr"], 1).fillna(0.0)
    macro_features["usdinr_mom_20d"] = _log_returns(macro_df["usd_inr"], 20).fillna(0.0)
    macro_features["us10y_yield_diff"] = macro_df["us_10y_yield"].diff().fillna(0.0)

    print("[4/4] Computing Panel Targets: Equations (1), (2), and (3) across h in {1, 5, 20}...")
    panel_records = []
    horizons = [1, 5, 20]
    bench_fwd: dict[int, pd.Series] = {h: _forward_sum(bench_ret, h) for h in horizons}

    for sector_name in sector_rets.columns:
        s_ret = sector_rets[sector_name]
        df_sec = pd.DataFrame(index=master_dates)
        df_sec["timestamp"] = master_dates
        df_sec["sector_id"] = sector_name
        df_sec["etf_ticker"] = SECTOR_UNIVERSE[sector_name]["etf"]
        df_sec["log_return"] = s_ret
        df_sec["ret_lag_1d"] = s_ret.shift(1)
        df_sec["ret_lag_5d"] = s_ret.rolling(5).sum()
        df_sec["ret_lag_20d"] = s_ret.rolling(20).sum()
        df_sec["realized_vol_20d"] = s_ret.rolling(20).std() * np.sqrt(252)

        for h in horizons:
            sec_fwd_h = _forward_sum(s_ret, h)
            y_excess = sec_fwd_h - bench_fwd[h]
            df_sec[f"y_excess_{h}d"] = y_excess
            df_sec[f"d_outperform_{h}d"] = np.where(y_excess > 0, 1, 0)

        df_sec = df_sec.join(macro_features, how="left")
        panel_records.append(df_sec)

    full_panel = pd.concat(panel_records, axis=0, ignore_index=True)
    full_panel = full_panel[full_panel["timestamp"] >= "2000-01-01"]
    full_panel.sort_values(["timestamp", "sector_id"], inplace=True)
    full_panel.dropna(subset=["realized_vol_20d", "bench_sma50_200_ratio", "y_excess_20d"], inplace=True)
    full_panel.reset_index(drop=True, inplace=True)

    raw_csv_path = Path(raw_dir) / "nse_sector_macro_raw.csv"
    parquet_path = Path(interim_dir) / "market_macro_panel.parquet"
    full_panel.to_csv(raw_csv_path, index=False)
    full_panel.to_parquet(parquet_path, index=False)

    print(f"\n[SUCCESS] Full 25-Year Market & Macro Panel Built!")
    print(f"  -> Date Range : {full_panel['timestamp'].min().date()} to {full_panel['timestamp'].max().date()}")
    print(f"  -> Total Rows : {len(full_panel):,} ({full_panel['sector_id'].nunique()} sectors x {len(full_panel)//8:,} trading days)")
    print(f"  -> Saved to   : {parquet_path}")
    return full_panel


if __name__ == "__main__":
    build_market_and_macro_pipeline()