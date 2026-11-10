import os
import sys
from datetime import datetime, timedelta

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from engine.store import get_store, init_store, register_asset, SOURCE_PATH

ASSET = "ml.fare_model"
MODEL_PATH = "models/fare_model.joblib"
TRAIN_START, TRAIN_END = "2024-01-01", "2024-01-15"
FIRST_MONITOR_DAY = datetime(2024, 1, 15)
FEATURES = ["trip_distance", "hour", "dow", "passenger_count", "pickup_zone_id"]


def load(start, end):
    sql = f"""
        SELECT trip_distance, hour(pickup_at) AS hour, dayofweek(pickup_at) AS dow,
               coalesce(passenger_count, 1) AS passenger_count,
               pickup_zone_id, fare_amount
        FROM main.stg_trips
        WHERE pickup_at >= '{start}' AND pickup_at < '{end}'
    """
    con = duckdb.connect(SOURCE_PATH, read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def psi(reference, current, bins=10):
    """Population Stability Index: how much has a distribution changed?"""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def train():
    print("Training the fare model (one time, about a minute)...")
    df = load(TRAIN_START, TRAIN_END)
    df = df.sample(min(300_000, len(df)), random_state=1)
    test = df.sample(frac=0.2, random_state=2)
    fit = df.drop(test.index)
    model = HistGradientBoostingRegressor(max_iter=150, random_state=1)
    model.fit(fit[FEATURES], fit["fare_amount"])
    base_mae = float(np.abs(model.predict(test[FEATURES]) - test["fare_amount"]).mean())
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "base_mae": base_mae,
                 "ref_distance": df["trip_distance"].sample(50_000, random_state=1).values},
                MODEL_PATH)
    print(f"Model saved. Baseline MAE: {base_mae:.2f} dollars")


def collect_ml(drift=False):
    init_store()
    register_asset(ASSET, "model", "fare_model", 3)
    if not os.path.exists(MODEL_PATH):
        train()
    bundle = joblib.load(MODEL_PATH)

    con = get_store()
    n = con.execute(
        "SELECT count(*) FROM events WHERE asset_id = ? AND metric = 'mae'", [ASSET]
    ).fetchone()[0]
    day = FIRST_MONITOR_DAY + timedelta(days=n % 17)   # each run = the next day
    batch = load(day, day + timedelta(days=1))

    if drift:   # simulate a problem: longer trips and a price increase
        batch["trip_distance"] = batch["trip_distance"] * 1.6
        batch["fare_amount"] = batch["fare_amount"] * 1.3

    pred = bundle["model"].predict(batch[FEATURES])
    mae = float(np.abs(pred - batch["fare_amount"]).mean())
    metrics = {
        "mae": mae,
        "mae_ratio": mae / bundle["base_mae"],
        "psi_trip_distance": psi(bundle["ref_distance"], batch["trip_distance"].values),
        "rows": float(len(batch)),
    }

    now = datetime.now()
    print(f"Batch day: {day.date()}  (drift={drift})")
    for name, value in metrics.items():
        bad = (name == "psi_trip_distance" and value > 0.25) or \
              (name == "mae_ratio" and value > 1.5)
        status = "fail" if bad else "ok"
        con.execute(
            "INSERT INTO events (ts, asset_id, module, metric, value, status, details) "
            "VALUES (?, ?, 'ml', ?, ?, ?, ?)",
            [now, ASSET, name, value, status, str(day.date())],
        )
        print(f"  {name:20} {value:10.3f}  [{status}]")
    con.close()


if __name__ == "__main__":
    collect_ml(drift="--drift" in sys.argv)