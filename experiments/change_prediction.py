import duckdb
import networkx as nx
import sqlglot
from sqlglot import exp

from experiments.common import save_results
from experiments.lab import SAMPLE_ROWS, Lab, fingerprints, propagate

SEED = 7000

# (id, table being edited, text to replace, new text)
CHANGES = [
    # --- breaking: a column that downstream tables use is renamed or dropped ---
    ("rename_fare_stg_trips", "stg_trips", "fare_amount, total_amount FROM", "fare_amount AS fare, total_amount FROM"),
    ("drop_total_stg_trips", "stg_trips", "fare_amount, total_amount FROM", "fare_amount FROM"),
    ("rename_borough_stg_zones", "stg_zones", "SELECT zone_id, borough FROM", "SELECT zone_id, borough AS boro FROM"),
    ("rename_revenue_daily", "daily_revenue", "SUM(total_amount) AS revenue", "SUM(total_amount) AS rev"),
    ("rename_trips_daily", "daily_revenue", "COUNT(*) AS trips", "COUNT(*) AS n_trips"),
    ("drop_hr_hourly", "hourly_demand", "EXTRACT(hour FROM pickup_at) AS hr, COUNT(*) AS hr_trips", "COUNT(*) AS hr_trips"),
    ("rename_fpm_features", "fare_features", "AS fare_per_mile", "AS fpm"),
    ("rename_brev_borough", "borough_revenue", "AS borough_rev", "AS brev"),
    # --- harmless: nobody uses the dropped or renamed column ---
    ("drop_unused_pax", "fare_features", "AS fare_per_mile, passenger_count AS pax FROM", "AS fare_per_mile FROM"),
    ("rename_leaf_avgfare", "avg_fare_by_pax", "AS avg_fare_pax", "AS avg_fare"),
    ("rename_leaf_exec", "exec_dashboard", "k.total_trips, b.borough_total", "k.total_trips AS trips_total, b.borough_total"),
    # --- harmless: a new column is added ---
    ("add_col_zones", "stg_zones", "SELECT zone_id, borough FROM", "SELECT zone_id, borough, zone_id * 2 AS zone_x2 FROM"),
    ("add_col_borough", "borough_revenue", "COUNT(*) AS borough_trips,", "COUNT(*) AS borough_trips, MAX(total_amount) AS borough_max,"),
    ("add_col_ops", "ops_dashboard", "SELECT hr, hr_trips FROM", "SELECT hr, hr_trips, hr_trips * 2 AS hr_x2 FROM"),
    # --- equivalent rewrites: same meaning, different text (hard cases) ---
    ("lowercase_kpi", "kpi_summary", "SELECT SUM(revenue) AS total_revenue, SUM(trips) AS total_trips", "select sum(revenue) as total_revenue, sum(trips) as total_trips"),
    ("reorder_predicates", "stg_trips", "fare_amount > 0 AND trip_distance > 0", "trip_distance > 0 AND fare_amount > 0"),
    ("reorder_in_list", "borough_revenue", "('Manhattan','Queens','Brooklyn','Bronx')", "('Bronx','Brooklyn','Queens','Manhattan')"),
    # --- logic changes: same columns, different numbers ---
    ("filter_fare", "stg_trips", "fare_amount > 0", "fare_amount > 5"),
    ("filter_distance", "stg_trips", "trip_distance > 0", "trip_distance > 1"),
    ("revenue_uses_fare", "daily_revenue", "SUM(total_amount) AS revenue", "SUM(fare_amount) AS revenue"),
    ("fewer_boroughs", "borough_revenue", "('Manhattan','Queens','Brooklyn','Bronx')", "('Manhattan','Queens')"),
    ("pax_range", "avg_fare_by_pax", "BETWEEN 1 AND 4", "BETWEEN 1 AND 2"),
    ("hour_shift", "hourly_demand", "EXTRACT(hour FROM pickup_at) AS hr", "EXTRACT(hour FROM pickup_at) + 1 AS hr"),
    ("left_join_zones", "trips_enriched", "JOIN stg_zones z", "LEFT JOIN stg_zones z"),
    ("kpi_double_count", "kpi_summary", "SUM(trips) AS total_trips", "SUM(trips) * 2 AS total_trips"),
    ("ml_scale", "ml_features", "AVG(fare_per_mile)", "AVG(fare_per_mile) * 1.5"),
]


def parts(sql):
    tree = sqlglot.parse_one(sql, read="duckdb")
    cols = {p.alias_or_name: p.unalias().sql(dialect="duckdb") for p in tree.expressions}
    rest = tree.copy()
    rest.set("expressions", [])          # everything except the SELECT list
    return {"cols": cols, "rest": rest.sql(dialect="duckdb")}


def used_columns(sql):
    tree = sqlglot.parse_one(sql, read="duckdb")
    return {c.name for c in tree.find_all(exp.Column)}


def schema_changed(lab, node, new_sql):
    return set(parts(lab.sql[node])["cols"]) != set(parts(new_sql)["cols"])


def predict(lab, node, new_sql):
    """Predict which tables would break or change, from the SQL text alone."""
    o, n = parts(lab.sql[node]), parts(new_sql)
    if o == n:
        return set()
    removed = set(o["cols"]) - set(n["cols"])
    changed = {c for c in o["cols"] if c in n["cols"] and o["cols"][c] != n["cols"][c]}
    rest_changed = o["rest"] != n["rest"]

    broken = set()                        # tables that would fail to build
    if removed:
        for child in lab.graph.successors(node):
            if removed & used_columns(lab.sql[child]):
                broken |= {child} | nx.descendants(lab.graph, child)

    values = {}                           # tables whose numbers would change
    if changed or rest_changed:
        values[node] = set(n["cols"]) if rest_changed else changed
        for name in lab.order:
            if name == node or name in lab.sources or name in broken:
                continue
            incoming = set()
            for p in lab.parents[name]:
                incoming |= values.get(p, set())
            if incoming:
                out = propagate(lab.sql[name], incoming)
                if out:
                    values[name] = out
    return broken | set(values)


def build_custom(lab, seed, sqls):
    trips = lab.pool.sample(SAMPLE_ROWS, random_state=seed).reset_index(drop=True)
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=1")
    con.register("trips_df", trips)
    con.register("zones_df", lab.zones)
    tables, failed = {}, set()
    for name in lab.order:
        try:
            if name in lab.sources:
                src = "trips_df" if name == "trips_raw" else "zones_df"
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM {src}")
            else:
                if any(p in failed for p in lab.parents[name]):
                    failed.add(name)
                    continue
                con.execute(f"CREATE TABLE {name} AS {sqls[name]}")
            tables[name] = con.execute(f"SELECT * FROM {name}").df()
        except Exception:
            failed.add(name)
    con.close()
    return tables, failed


def truth_affected(lab, node, new_sql):
    """Run old and new pipelines. A table is affected if it fails, or if its
    old columns now hold different values."""
    clean, _ = build_custom(lab, SEED, lab.sql)
    sqls = dict(lab.sql)
    sqls[node] = new_sql
    new, failed = build_custom(lab, SEED, sqls)
    changed = set()
    for name, df in new.items():
        cols = [c for c in clean[name].columns if c in df.columns]
        if not cols:
            changed.add(name)
            continue
        if fingerprints({"x": clean[name][cols]})["x"] != fingerprints({"x": df[cols]})["x"]:
            changed.add(name)
    return failed | changed, failed


def main():
    lab = Lab()
    rows = []
    for cid, node, old, new in CHANGES:
        new_sql = lab.sql[node].replace(old, new)
        assert new_sql != lab.sql[node], f"change {cid} did not modify the SQL"
        truth, failed = truth_affected(lab, node, new_sql)
        everything = {node} | nx.descendants(lab.graph, node)
        rows.append(dict(
            id=cid, truth=truth, failed=failed,
            ours=predict(lab, node, new_sql),
            any=everything,
            schema=everything if schema_changed(lab, node, new_sql) else set()))

    risky = [bool(r["truth"]) for r in rows]
    n_risky, n_safe = sum(risky), len(rows) - sum(risky)
    print(f"{len(rows)} changes: {n_risky} truly risky, {n_safe} truly safe\n")
    methods = {"ours: AST + column lineage": "ours",
               "baseline: any change is risky": "any",
               "baseline: schema diff only": "schema"}
    results = []
    for label, key in methods.items():
        flag = [bool(r[key]) for r in rows]
        tp = sum(f and t for f, t in zip(flag, risky))
        fp = sum(f and not t for f, t in zip(flag, risky))
        fn = sum((not f) and t for f, t in zip(flag, risky))
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        stp = sum(len(r[key] & r["truth"]) for r in rows)
        sfp = sum(len(r[key] - r["truth"]) for r in rows)
        sfn = sum(len(r["truth"] - r[key]) for r in rows)
        results += [
            (label, "review_precision", p), (label, "review_recall", rec),
            (label, "review_f1", f1), (label, "false_alarm_rate", fp / max(n_safe, 1)),
            (label, "affected_set_precision", stp / (stp + sfp) if stp + sfp else 0.0),
            (label, "affected_set_recall", stp / (stp + sfn) if stp + sfn else 0.0),
        ]
        print(f"{label:32} precision {p:.2f}  recall {rec:.2f}  f1 {f1:.2f}  "
              f"false alarms {fp}/{n_safe}  missed {fn}/{n_risky}")

    print("\nPer change:  id | what really happened | true size | ours size | ours right?")
    for r in rows:
        kind = "breaks build" if r["failed"] else ("changes values" if r["truth"] else "safe")
        ok = bool(r["ours"]) == bool(r["truth"])
        print(f"{r['id']:26} {kind:15} {len(r['truth']):3} {len(r['ours']):3}   {'yes' if ok else 'NO'}")
    save_results("change_prediction", results, len(rows))
    print("\nSaved to the metadata store.")


if __name__ == "__main__":
    main()