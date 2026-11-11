from datetime import datetime

from engine.store import get_store, init_store
from experiments.lab import outputs_of, propagate

# Business value (dollars per day) of each table. This is the "ground truth" model
# for the impact experiment. It is DIFFERENT from the 1-3 criticality scores the
# system uses, so the system has to estimate it, not copy it.
VALUE_PER_DAY = {
    "trips_raw": 1000, "zones_raw": 300, "stg_trips": 1000, "stg_zones": 300,
    "trips_enriched": 4000, "daily_revenue": 15000, "borough_revenue": 12000,
    "hourly_demand": 6000, "fare_features": 8000, "avg_fare_by_pax": 2000,
    "kpi_summary": 20000, "borough_total": 9000, "exec_dashboard": 50000,
    "ops_dashboard": 9000, "ml_features": 30000,
}


def save_results(experiment, results, n):
    init_store()
    con = get_store()
    con.execute("CREATE TABLE IF NOT EXISTS experiment_results (experiment VARCHAR, "
                "method VARCHAR, metric VARCHAR, value DOUBLE, n INTEGER, ts TIMESTAMP)")
    con.execute("DELETE FROM experiment_results WHERE experiment = ?", [experiment])
    now = datetime.now()
    for method, metric, value in results:
        con.execute("INSERT INTO experiment_results VALUES (?,?,?,?,?,?)",
                    [experiment, method, metric, float(value), int(n), now])
    con.close()


def root_outputs(lab, root):
    if root in lab.sources:
        return set(lab.source_cols[root])
    return set(outputs_of(lab.sql[root]))


def blast_from_root(lab, root):
    """Column-level blast radius, assuming every column of the root is bad."""
    affected = {root: root_outputs(lab, root)}
    for name in lab.order:
        if name == root or name in lab.sources:
            continue
        incoming = set()
        for p in lab.parents[name]:
            incoming |= affected.get(p, set())
        if incoming:
            out = propagate(lab.sql[name], incoming)
            if out:
                affected[name] = out
    return set(affected)