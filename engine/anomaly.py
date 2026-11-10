import pandas as pd
from engine.store import get_store, init_store

WINDOW = 10        # compare with the last 10 values
MIN_HISTORY = 5    # need at least 5 past values before judging
Z_LIMIT = 3.0      # more than 3 standard deviations away = anomaly


def detect_anomalies():
    init_store()
    con = get_store()
    df = con.execute("""
        SELECT event_id, ts, asset_id, module, metric, value
        FROM events
        WHERE value IS NOT NULL
          AND module IN ('data_quality', 'streaming', 'pipeline', 'ml')
        ORDER BY ts, event_id
    """).df()

    found = []
    for (asset, module, metric), g in df.groupby(["asset_id", "module", "metric"]):
        g = g.sort_values(["ts", "event_id"]).copy()
        past = g["value"].shift(1).rolling(WINDOW, min_periods=MIN_HISTORY)
        g["expected"] = past.mean()
        std = past.std()
        # floor the spread, so a flat history does not alarm on tiny noise
        spread = std.where(std > 0.01 * g["expected"].abs(), 0.01 * g["expected"].abs())
        spread = spread.clip(lower=1e-9)
        g["zscore"] = (g["value"] - g["expected"]) / spread
        bad = g[g["zscore"].abs() > Z_LIMIT]
        found.append(bad[["event_id", "ts", "asset_id", "module", "metric",
                          "value", "expected", "zscore"]])

    con.execute("DELETE FROM anomalies")
    if found:
        out = pd.concat(found)
        if len(out):
            con.register("anom_df", out)
            con.execute("INSERT INTO anomalies SELECT * FROM anom_df")
    n = con.execute("SELECT count(*) FROM anomalies").fetchone()[0]
    con.close()
    print(f"Anomalies found: {n}")


if __name__ == "__main__":
    detect_anomalies()