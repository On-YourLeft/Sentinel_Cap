from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.joblib"
LOG_FILE = PROJECT_ROOT / "logs" / "prediction_inputs.jsonl"


def run_drift_check(z_threshold: float = 2.0) -> None:
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model artifact not found at {MODEL_PATH}. Run train.py first.")
        return
    if not LOG_FILE.exists():
        print(f"[INFO] No logged predictions yet at {LOG_FILE}. Send requests to POST /predict first.")
        return

    bundle = joblib.load(MODEL_PATH)
    train_stats: dict = bundle["metadata"]["train_stats"]
    records = [json.loads(line)["inputs"] for line in open(LOG_FILE, encoding="utf-8") if line.strip()]
    if not records:
        print("[INFO] Prediction log is empty.")
        return

    live_df = pd.DataFrame(records)
    print(f"\n=== SENTINEL-CAP INPUT DRIFT REPORT (n={len(live_df)} live requests) ===")
    print(f"{'Feature':<30} | {'Train Mean':>10} | {'Live Mean':>10} | {'Z-Shift':>8} | {'Status'}")
    print("-" * 78)
    for feat, stats in train_stats.items():
        if feat in live_df.columns:
            t_mean, t_std = stats["mean"], stats["std"]
            l_mean = float(live_df[feat].mean())
            z_shift = abs(l_mean - t_mean) / max(t_std, 1e-6)
            status = "DRIFT ALERT" if z_shift > z_threshold else "OK"
            print(f"{feat:<30} | {t_mean:>10.4f} | {l_mean:>10.4f} | {z_shift:>7.2f}s | {status}")


if __name__ == "__main__":
    run_drift_check()
