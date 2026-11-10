import time
from datetime import datetime

import duckdb
from engine.store import get_store, init_store, SOURCE_PATH

PRICE_PER_SECOND = 0.01   # SIMULATED price: like a $36/hour warehouse

# (job name, table it reads, SQL). These act as "consumers" of your data.
QUERIES = [
    ("revenue_dashboard", "main.mart_daily_revenue",
     "SELECT trip_date, revenue FROM main.mart_daily_revenue ORDER BY 1"),
    ("zone_report", "main.mart_zone_trips",
     "SELECT borough, sum(trips) FROM main.mart_zone_trips GROUP BY 1"),
    ("analyst_adhoc_scan", "main.stg_trips",
     "SELECT date_trunc('hour', pickup_at) h, avg(fare_amount), count(*) "
     "FROM main.stg_trips GROUP BY 1"),
]
RUNS_PER_QUERY = 3


def collect_cost():
    init_store()
    store = get_store()

    # 1) Cost of pipeline steps, taken from runtimes already logged by Dagster
    last = store.execute(
        "SELECT coalesce(max(ts), TIMESTAMP '1970-01-01') FROM costs "
        "WHERE job_name LIKE 'pipeline.%'"
    ).fetchone()[0]
    rows = store.execute(
        "SELECT ts, asset_id, value FROM events "
        "WHERE module = 'pipeline' AND metric = 'runtime_seconds' AND ts > ?",
        [last],
    ).fetchall()
    for ts, asset_id, seconds in rows:
        store.execute(
            "INSERT INTO costs (ts, job_name, asset_id, runtime_seconds, cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            [ts, asset_id, asset_id, seconds, seconds * PRICE_PER_SECOND],
        )
    print(f"Added {len(rows)} pipeline cost rows")

    # 2) Cost of consumer queries, measured by running them
    source = duckdb.connect(SOURCE_PATH, read_only=True)
    for job, table, sql in QUERIES:
        for _ in range(RUNS_PER_QUERY):
            start = time.time()
            source.execute(sql).fetchall()
            seconds = time.time() - start
            store.execute(
                "INSERT INTO costs (ts, job_name, asset_id, runtime_seconds, cost_usd) "
                "VALUES (?, ?, ?, ?, ?)",
                [datetime.now(), f"query.{job}", table, seconds,
                 seconds * PRICE_PER_SECOND],
            )
        print(f"query.{job:22} ran {RUNS_PER_QUERY}x on {table}")
    source.close()
    store.close()


if __name__ == "__main__":
    collect_cost()