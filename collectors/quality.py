import duckdb
from engine.store import get_store, init_store, SOURCE_PATH

# (asset, metric name, SQL that returns one number)
CHECKS = [
    ("raw.yellow_trips", "row_count",
     "SELECT count(*) FROM raw.yellow_trips"),
    ("raw.yellow_trips", "null_rate_passenger_count",
     "SELECT avg(CASE WHEN passenger_count IS NULL THEN 1.0 ELSE 0.0 END) FROM raw.yellow_trips"),
    ("main.stg_trips", "row_count",
     "SELECT count(*) FROM main.stg_trips"),
    ("main.stg_trips", "avg_fare",
     "SELECT avg(fare_amount) FROM main.stg_trips"),
    ("main.stg_trips", "duplicate_rate",
     """SELECT 1.0 - count(DISTINCT (pickup_at, dropoff_at, pickup_zone_id,
        dropoff_zone_id, fare_amount, total_amount)) * 1.0 / count(*)
        FROM main.stg_trips"""),
    ("main.mart_daily_revenue", "row_count",
     "SELECT count(*) FROM main.mart_daily_revenue"),
    ("main.mart_zone_trips", "row_count",
     "SELECT count(*) FROM main.mart_zone_trips"),
]


def status_for(metric, value):
    if value is None:
        return "fail"
    if metric == "row_count" and value == 0:
        return "fail"
    if metric == "duplicate_rate" and value > 0.01:
        return "fail"
    return "ok"


def collect_quality():
    init_store()
    source = duckdb.connect(SOURCE_PATH, read_only=True)
    store = get_store()
    for asset_id, metric, sql in CHECKS:
        value = source.execute(sql).fetchone()[0]
        value = None if value is None else float(value)
        status = status_for(metric, value)
        store.execute(
            "INSERT INTO events (asset_id, module, metric, value, status) VALUES (?, ?, ?, ?, ?)",
            [asset_id, "data_quality", metric, value, status],
        )
        print(f"{asset_id:28} {metric:28} {value}  [{status}]")
    source.close()
    store.close()


if __name__ == "__main__":
    collect_quality()