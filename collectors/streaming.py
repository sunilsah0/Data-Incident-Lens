import sys
from datetime import datetime

import numpy as np
from engine.store import get_store, init_store, register_asset

ASSET = "stream.trip_events"
N_EVENTS = 20000
WATERMARK_SECONDS = 60   # an event arriving later than this counts as "late"


def simulate_lags(incident=False):
    rng = np.random.default_rng()
    lag = rng.exponential(scale=2.0, size=N_EVENTS)   # normal: about 2 seconds
    if incident:
        slow = rng.random(N_EVENTS) < 0.30            # 30% of events get stuck
        lag[slow] += rng.uniform(40, 180, size=int(slow.sum()))
    return lag


def collect_streaming(incident=False):
    init_store()
    register_asset(ASSET, "stream", "trip_events", 2)

    lag = simulate_lags(incident)
    metrics = {
        "avg_lag_seconds": float(lag.mean()),
        "p95_lag_seconds": float(np.percentile(lag, 95)),
        "late_event_rate": float((lag > WATERMARK_SECONDS).mean()),
    }

    now = datetime.now()
    con = get_store()
    for name, value in metrics.items():
        bad = (name == "p95_lag_seconds" and value > 30) or \
              (name == "late_event_rate" and value > 0.05)
        status = "fail" if bad else "ok"
        con.execute(
            "INSERT INTO events (ts, asset_id, module, metric, value, status) "
            "VALUES (?, ?, 'streaming', ?, ?, ?)",
            [now, ASSET, name, value, status],
        )
        print(f"{name:20} {value:10.3f}  [{status}]")
    con.close()


if __name__ == "__main__":
    collect_streaming(incident="--incident" in sys.argv)