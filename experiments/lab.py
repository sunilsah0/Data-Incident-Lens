import os

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp

from engine.store import SOURCE_PATH

POOL_FILE = "data/lab_pool.parquet"
ZONES_FILE = "data/lab_zones.parquet"
SAMPLE_ROWS = 6000
Z_LIMIT = 4.0

# (name, parents, SQL, business criticality). Sources have no SQL.
NODES = [
    ("trips_raw", [], None, 1),
    ("zones_raw", [], None, 1),
    ("stg_trips", ["trips_raw"],
     "SELECT trip_id, pickup_at, pickup_zone_id, passenger_count, trip_distance, "
     "fare_amount, total_amount FROM trips_raw WHERE fare_amount > 0 AND trip_distance > 0", 2),
    ("stg_zones", ["zones_raw"],
     "SELECT zone_id, borough FROM zones_raw WHERE borough IS NOT NULL", 2),
    ("trips_enriched", ["stg_trips", "stg_zones"],
     "SELECT t.trip_id, t.pickup_at, t.passenger_count, t.trip_distance, t.fare_amount, "
     "t.total_amount, z.borough FROM stg_trips t JOIN stg_zones z "
     "ON t.pickup_zone_id = z.zone_id", 2),
    ("daily_revenue", ["stg_trips"],
     "SELECT CAST(pickup_at AS DATE) AS trip_date, COUNT(*) AS trips, "
     "SUM(total_amount) AS revenue FROM stg_trips GROUP BY CAST(pickup_at AS DATE)", 2),
    ("borough_revenue", ["trips_enriched"],
     "SELECT borough, COUNT(*) AS borough_trips, SUM(total_amount) AS borough_rev "
     "FROM trips_enriched WHERE borough IN ('Manhattan','Queens','Brooklyn','Bronx') "
     "GROUP BY borough", 2),
    ("hourly_demand", ["stg_trips"],
     "SELECT EXTRACT(hour FROM pickup_at) AS hr, COUNT(*) AS hr_trips FROM stg_trips "
     "GROUP BY EXTRACT(hour FROM pickup_at)", 2),
    ("fare_features", ["trips_enriched"],
     "SELECT trip_id, fare_amount / trip_distance AS fare_per_mile, "
     "passenger_count AS pax FROM trips_enriched", 2),
    ("avg_fare_by_pax", ["stg_trips"],
     "SELECT passenger_count AS pax_group, AVG(fare_amount) AS avg_fare_pax FROM stg_trips "
     "WHERE passenger_count BETWEEN 1 AND 4 GROUP BY passenger_count", 1),
    ("kpi_summary", ["daily_revenue"],
     "SELECT SUM(revenue) AS total_revenue, SUM(trips) AS total_trips FROM daily_revenue", 2),
    ("borough_total", ["borough_revenue"],
     "SELECT SUM(borough_rev) AS borough_total FROM borough_revenue", 1),
    ("exec_dashboard", ["kpi_summary", "borough_total"],
     "SELECT k.total_revenue, k.total_trips, b.borough_total FROM kpi_summary k "
     "CROSS JOIN borough_total b", 3),
    ("ops_dashboard", ["hourly_demand"],
     "SELECT hr, hr_trips FROM hourly_demand WHERE hr_trips > 0", 2),
    ("ml_features", ["fare_features"],
     "SELECT AVG(fare_per_mile) AS mean_fpm, COUNT(*) AS n_rows FROM fare_features", 3),
]

DATA_FAULTS = [
    dict(id="dup_trips", node="trips_raw", kind="duplicate", col=None),
    dict(id="missing_trips", node="trips_raw", kind="drop", col=None),
    dict(id="null_fare", node="trips_raw", kind="nulls", col="fare_amount"),
    dict(id="null_passengers", node="trips_raw", kind="nulls", col="passenger_count"),
    dict(id="null_zone", node="trips_raw", kind="nulls", col="pickup_zone_id"),
    dict(id="unit_distance", node="trips_raw", kind="scale", col="trip_distance", factor=1.6),
    dict(id="unit_total", node="trips_raw", kind="scale", col="total_amount", factor=1.5),
    dict(id="dup_zones", node="zones_raw", kind="duplicate", col=None),
    dict(id="null_borough", node="zones_raw", kind="nulls", col="borough"),
]
LOGIC_FAULTS = [
    dict(id="bad_filter_trips", node="stg_trips", kind="logic",
         old="fare_amount > 0 AND trip_distance > 0", new="fare_amount > 12 AND trip_distance > 0"),
    dict(id="bad_filter_zones", node="stg_zones", kind="logic",
         old="borough IS NOT NULL", new="borough <> 'Manhattan'"),
    dict(id="wrong_revenue_col", node="daily_revenue", kind="logic",
         old="SUM(total_amount)", new="SUM(fare_amount)"),
    dict(id="borough_unit_bug", node="borough_revenue", kind="logic",
         old="SUM(total_amount)", new="SUM(total_amount) / 100"),
    dict(id="hour_bucket_bug", node="hourly_demand", kind="logic",
         old="EXTRACT(hour FROM pickup_at)", new="EXTRACT(hour FROM pickup_at) / 2"),
    dict(id="distance_unit_bug", node="fare_features", kind="logic",
         old="fare_amount / trip_distance", new="fare_amount / (trip_distance * 1.609)"),
    dict(id="median_not_mean", node="avg_fare_by_pax", kind="logic",
         old="AVG(fare_amount)", new="MEDIAN(fare_amount)"),
    dict(id="double_count_trips", node="kpi_summary", kind="logic",
         old="SUM(trips)", new="SUM(trips) * 2"),
    dict(id="feature_scale_bug", node="ml_features", kind="logic",
         old="AVG(fare_per_mile)", new="AVG(fare_per_mile) * 1.5"),
]
ALL_FAULTS = DATA_FAULTS + LOGIC_FAULTS


def load_pool():
    if not os.path.exists(POOL_FILE):
        con = duckdb.connect(SOURCE_PATH, read_only=True)
        con.execute("SELECT setseed(0.42)")
        con.execute(f"""COPY (
            SELECT row_number() OVER () AS trip_id, pickup_at, pickup_zone_id,
                   passenger_count, trip_distance, fare_amount, total_amount
            FROM (SELECT * FROM main.stg_trips ORDER BY random() LIMIT 100000)
        ) TO '{POOL_FILE}' (FORMAT PARQUET)""")
        con.execute(f"""COPY (SELECT LocationID AS zone_id, Borough AS borough
                        FROM raw.taxi_zones) TO '{ZONES_FILE}' (FORMAT PARQUET)""")
        con.close()
    pool = duckdb.sql(f"SELECT * FROM read_parquet('{POOL_FILE}')").df()
    zones = duckdb.sql(f"SELECT * FROM read_parquet('{ZONES_FILE}')").df()
    return pool, zones


def apply_data_fault(fault, df, rng):
    df = df.copy()
    kind = fault["kind"]
    if kind == "duplicate":
        extra = df.sample(frac=0.25, random_state=int(rng.integers(1_000_000)))
        df = pd.concat([df, extra], ignore_index=True)
    elif kind == "drop":
        df = df.sample(frac=0.6, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    elif kind == "nulls":
        col = fault["col"]
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("float64")
        df.loc[rng.random(len(df)) < 0.3, col] = np.nan
    elif kind == "scale":
        df[fault["col"]] = df[fault["col"]] * fault["factor"]
    return df


class Lab:
    def __init__(self):
        self.pool, self.zones = load_pool()
        self.order = [n[0] for n in NODES]
        self.parents = {n[0]: n[1] for n in NODES}
        self.sql = {n[0]: n[2] for n in NODES}
        self.crit = {n[0]: n[3] for n in NODES}
        self.sources = {n[0] for n in NODES if n[2] is None}
        self.source_cols = {"trips_raw": list(self.pool.columns),
                            "zones_raw": list(self.zones.columns)}
        self.graph = nx.DiGraph()
        for name in self.order:
            self.graph.add_node(name)
            for p in self.parents[name]:
                self.graph.add_edge(p, name)

    def build(self, seed, fault=None):
        """Build all tables for one 'day' of data, optionally with a fault."""
        trips = self.pool.sample(SAMPLE_ROWS, random_state=seed).reset_index(drop=True)
        zones = self.zones.copy()
        sqls = dict(self.sql)
        if fault is not None:
            if fault["kind"] == "logic":
                new_sql = sqls[fault["node"]].replace(fault["old"], fault["new"])
                assert new_sql != sqls[fault["node"]], "fault did not change the SQL"
                sqls[fault["node"]] = new_sql
            else:
                rng = np.random.default_rng(seed + 12345)
                if fault["node"] == "trips_raw":
                    trips = apply_data_fault(fault, trips, rng)
                else:
                    zones = apply_data_fault(fault, zones, rng)
        con = duckdb.connect(":memory:")
        con.execute("PRAGMA threads=1")
        con.register("trips_df", trips)
        con.register("zones_df", zones)
        con.execute("CREATE TABLE trips_raw AS SELECT * FROM trips_df")
        con.execute("CREATE TABLE zones_raw AS SELECT * FROM zones_df")
        tables = {}
        for name in self.order:
            if sqls[name] is not None:
                con.execute(f"CREATE TABLE {name} AS {sqls[name]}")
            tables[name] = con.execute(f"SELECT * FROM {name}").df()
        con.close()
        return tables


# ---------- monitoring ----------
def table_metrics(tables):
    m = {}
    for name, df in tables.items():
        m[(name, "rows")] = float(len(df))
        for col in df.columns:
            s = df[col]
            m[(name, f"{col}.null")] = float(s.isna().mean()) if len(df) else 0.0
            numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            if numeric and col != "trip_id" and s.notna().any():
                m[(name, f"{col}.mean")] = float(s.mean())
    return m


def baseline_stats(lab, seeds):
    runs = [table_metrics(lab.build(s)) for s in seeds]
    stats = {}
    for key in runs[0]:
        vals = np.array([r[key] for r in runs if key in r])
        stats[key] = (float(vals.mean()), float(vals.std(ddof=1)))
    return stats


def detect(metrics, stats, z_limit=Z_LIMIT):
    alerts = []
    for key, value in metrics.items():
        if key not in stats:
            continue
        mean, std = stats[key]
        floor = 0.01 if key[1].endswith(".null") else 0.01 * abs(mean)
        z = (value - mean) / max(std, floor, 1e-9)
        if abs(z) > z_limit:
            alerts.append((key[0], key[1], z))
    return alerts


def fingerprints(tables):
    """One number per table. Different number = the table's content changed."""
    out = {}
    for name, df in tables.items():
        d = df.round(6)
        d = d.sort_values(list(d.columns)).reset_index(drop=True)
        out[name] = int(pd.util.hash_pandas_object(d, index=False).values.sum())
    return out


# ---------- column-level lineage ----------
def outputs_of(sql):
    return [p.alias_or_name for p in sqlglot.parse_one(sql, read="duckdb").expressions]


def propagate(sql, affected):
    """Which output columns of this query are hit, if these input columns are bad?"""
    select = sqlglot.parse_one(sql, read="duckdb")
    row_cols = set()
    for clause in ("where", "group", "having"):
        node = select.args.get(clause)
        if node is not None:
            row_cols |= {c.name for c in node.find_all(exp.Column)}
    for join in select.args.get("joins") or []:
        on = join.args.get("on")
        if on is not None:
            row_cols |= {c.name for c in on.find_all(exp.Column)}
    outs = {p.alias_or_name: {c.name for c in p.find_all(exp.Column)}
            for p in select.expressions}
    if row_cols & affected:        # bad column decides which rows survive: all outputs hit
        return set(outs)
    return {name for name, cols in outs.items() if cols & affected}


def predict_affected(lab, fault):
    """Column-level prediction of which tables a fault damages."""
    node = fault["node"]
    affected = {}
    if fault["kind"] == "logic":
        affected[node] = set(outputs_of(lab.sql[node]))
    else:
        cols = [fault["col"]] if fault.get("col") else lab.source_cols[node]
        affected[node] = set(cols)
    for name in lab.order:
        if name == node or name in lab.sources:
            continue
        incoming = set()
        for p in lab.parents[name]:
            incoming |= affected.get(p, set())
        if incoming:
            out = propagate(lab.sql[name], incoming)
            if out:
                affected[name] = out
    return set(affected)