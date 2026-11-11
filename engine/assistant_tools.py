import json

import duckdb
import networkx as nx

from engine.store import read_path


def _q(sql, params=None):
    con = duckdb.connect(read_path(), read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def _graph():
    e = _q("SELECT upstream_id, downstream_id FROM lineage_edges")
    g = nx.DiGraph()
    g.add_edges_from(zip(e.upstream_id, e.downstream_id))
    return g


def all_asset_ids():
    a = _q("SELECT asset_id FROM assets UNION SELECT DISTINCT asset_id FROM events")
    return a["asset_id"].dropna().tolist()


def resolve_asset(name):
    ids = all_asset_ids()
    if name in ids:
        return name
    hits = [x for x in ids if x.split(".")[-1] == name.split(".")[-1]]
    return hits[0] if hits else None


def list_incidents(limit=5):
    df = _q("SELECT incident_id, severity, root_cause_asset, title, opened_at "
            "FROM incidents ORDER BY severity DESC LIMIT ?", [int(limit)])
    return df.to_dict("records")


def explain_incident(incident_id):
    inc = _q("SELECT * FROM incidents WHERE incident_id = ?", [int(incident_id)])
    if inc.empty:
        return {"error": f"No incident with id {incident_id}."}
    row = inc.iloc[0]
    d = json.loads(row["details"])
    ids = d["assets_with_alerts"]
    marks = ",".join("?" for _ in ids)
    an = _q(f"SELECT asset_id, module, metric, value, expected, zscore FROM anomalies "
            f"WHERE asset_id IN ({marks}) ORDER BY abs(zscore) DESC LIMIT 8", ids)
    return {"incident_id": int(row["incident_id"]), "title": row["title"],
            "severity": float(row["severity"]), "root_cause": row["root_cause_asset"],
            "why": d["ranking"][:3], "affected_assets": d["affected"],
            "strongest_alerts": an.round(3).to_dict("records")}


def blast_radius(asset):
    a = resolve_asset(asset)
    if a is None:
        return {"error": f"Unknown asset '{asset}'."}
    g = _graph()
    c = _q("SELECT asset_id, criticality FROM assets")
    crit = dict(zip(c.asset_id, c.criticality))
    down = sorted(nx.descendants(g, a)) if a in g else []
    up = sorted(nx.ancestors(g, a)) if a in g else []
    return {"asset": a, "downstream": [{"asset": x, "criticality": crit.get(x)} for x in down],
            "upstream": up}


def asset_health(asset):
    a = resolve_asset(asset)
    if a is None:
        return {"error": f"Unknown asset '{asset}'."}
    ev = _q("SELECT module, metric, value, status FROM events WHERE asset_id = ? "
            "QUALIFY row_number() OVER (PARTITION BY module, metric "
            "ORDER BY ts DESC, event_id DESC) = 1", [a])
    n = _q("SELECT count(*) AS n FROM anomalies WHERE asset_id = ?", [a])["n"].iloc[0]
    return {"asset": a, "latest_checks": ev.round(4).to_dict("records"),
            "anomalies_seen": int(n)}


def cost_summary():
    df = _q("SELECT job_name, sum(cost_usd) AS cost_usd, count(*) AS runs "
            "FROM costs GROUP BY 1 ORDER BY 2 DESC")
    return {"total_cost_usd": round(float(df["cost_usd"].sum()), 2),
            "top_jobs": df.head(5).round(4).to_dict("records"),
            "note": "Simulated price per compute-second; runtimes are real."}


def experiment_results(name=None):
    df = _q("SELECT experiment, method, metric, value, n FROM experiment_results "
            "ORDER BY 1, 2, 3")
    if name:
        df = df[df["experiment"].str.contains(name.replace(" ", "_"), case=False)]
    return df.round(3).to_dict("records")


# ---------- recommendations (rule-based, never LLM-generated) ----------
RULES = {
    "psi_trip_distance": "alert when PSI is above 0.25 (big input shift)",
    "mae_ratio": "alert when the error is more than 1.5x the training baseline",
    "mae": "alert when the error is more than 1.5x the training baseline",
    "p95_lag_seconds": "alert when p95 event lag is above 30 seconds",
    "late_event_rate": "alert when more than 5% of events are late",
    "stale_answer_rate": "alert when more than 5% of answers use old document versions",
    "unauthorized_answer_rate": "alert when more than 5% of answers use restricted sources",
    "ungrounded_answer_rate": "alert when more than 5% of answers are unsupported",
    "access_violation_rate": "alert when more than 5% of accesses are denied",
    "pii_unauthorized_accesses": "alert when more than 5 accesses to personal data are denied",
}


def _monitor_rule(metric):
    rule = RULES.get(metric, "alert when it is more than 3 standard deviations "
                             "from the average of its last 10 values")
    return f"`{metric}`: {rule}."


def _advice(metric):
    m = metric.lower()
    if m == "row_count":
        return "Check the upstream load or source extract for missing or duplicated data."
    if m.startswith("null_rate"):
        return "Find which upstream change started producing nulls; check the source schema."
    if m == "duplicate_rate":
        return "Look for a re-run or double load, then deduplicate on the business key."
    if "lag" in m or "late_event" in m:
        return "Check consumer capacity and backpressure; scale the consumer or fix the slow producer."
    if m.startswith("psi"):
        return "Input data has shifted. Find the source change, then decide whether to retrain."
    if m.startswith("mae"):
        return "Model error is up. Compare with the last good batch and consider retraining."
    if m == "stale_answer_rate":
        return "Refresh the index and retire superseded document versions."
    if m == "unauthorized_answer_rate":
        return "The access filter is failing. Re-enable it before serving more answers."
    if m == "ungrounded_answer_rate":
        return "Turn the abstain guard back on so the system says it does not know."
    if m.startswith("access_violation") or m.startswith("pii_unauthorized"):
        return "Revoke the offending access and review the access log."
    if m == "runtime_seconds":
        return "A pipeline step is running unusually long. Check data volume and locks."
    return "Compare with recent changes to the pipeline or its inputs."

def _dbt_test(asset, module, metric, expected):
    if module != "data_quality" or "." not in asset:
        return None
    schema, name = asset.split(".", 1)
    if schema == "main":
        ref = "{{ ref('" + name + "') }}"
    elif schema == "raw":
        ref = "{{ source('raw', '" + name + "') }}"
    else:
        return None
    if metric == "row_count":
        body = f"select 1 from (select count(*) as n from {ref}) where n < {int(expected * 0.5)}"
    elif metric.startswith("null_rate_"):
        col = metric[len("null_rate_"):]
        body = (f"select 1 from (select avg(case when {col} is null then 1.0 else 0.0 end) "
                f"as r from {ref}) where r > {expected * 2 + 0.01:.4f}")
    else:
        return None
    return {"file": f"pipelines/tests/assert_{name}_{metric}.sql", "sql": body}


def recommend_actions(incident_id):
    inc = _q("SELECT * FROM incidents WHERE incident_id = ?", [int(incident_id)])
    if inc.empty:
        return {"error": f"No incident with id {incident_id}."}
    row = inc.iloc[0]
    root = row["root_cause_asset"]
    d = json.loads(row["details"])
    an = _q("SELECT module, metric, value, expected FROM anomalies WHERE asset_id = ? "
            "ORDER BY abs(zscore) DESC LIMIT 3", [root])
    affected = d["affected"]
    steps = [f"Recover: rebuild only the {len(affected)} affected assets "
             f"({', '.join(affected)}), not the whole pipeline. Re-run the monitors "
             f"afterwards to confirm the alerts clear."]
    grouped, tests, rules = {}, [], []
    for _, r in an.iterrows():
        fact = f"`{r['metric']}` is {float(r['value']):.3g} (expected about {float(r['expected']):.3g})"
        grouped.setdefault(_advice(r["metric"]), []).append(fact)
        t = _dbt_test(root, r["module"], r["metric"], float(r["expected"]))
        if t:
            tests.append(t)
        else:
            rules.append(_monitor_rule(r["metric"]))
    for advice, facts in grouped.items():
        steps.append("; ".join(facts) + ". " + advice)
    steps.append("Verify: compare row counts and key totals of the rebuilt tables with the last good run.")
    return {"incident_id": int(row["incident_id"]), "root_cause": root,
            "steps": steps, "prevention_tests": tests,
            "monitor_rules": sorted(set(rules))}