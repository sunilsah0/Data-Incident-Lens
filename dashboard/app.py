import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import pandas as pd
import streamlit as st

from engine import assistant
from engine.store import read_path


st.set_page_config(page_title="SunilDataIncidentLens", layout="wide")


@st.cache_data(ttl=10)
def query(sql):
    con = duckdb.connect(read_path(), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


st.title("SunilDataIncidentLens")
st.caption("Data and AI reliability control tower")

try:
    latest = query("""
        SELECT asset_id, metric, value, status, ts FROM events
        WHERE module = 'data_quality'
        QUALIFY row_number() OVER (PARTITION BY asset_id, metric ORDER BY ts DESC) = 1
    """)
    assets = query("SELECT * FROM assets")
    edges = query("SELECT * FROM lineage_edges")
    pipeline = query("SELECT ts, asset_id, value, status, details FROM events "
                     "WHERE module = 'pipeline' ORDER BY ts DESC")
    anoms = query("SELECT ts, asset_id, module, metric, value, expected, zscore "
            "FROM anomalies ORDER BY ts DESC")
except Exception as e:
    st.error(f"Could not read the metadata store. Is Dagster writing to it right now? Refresh in a few seconds.\n\n{e}")
    st.stop()

(tab_overview, tab_quality, tab_pipeline, tab_stream, tab_cost,
 tab_ml, tab_ai, tab_gov, tab_inc, tab_assist, tab_exp) = st.tabs(
    ["Overview", "Data Quality", "Pipeline Monitoring", "Real-Time Analytics",
     "Cost Intelligence", "ML Monitoring", "AI Reliability", "Governance & Security",
     "Incident Management", "AI Engineering Assistant", "Experiments"]
)
# ---------- Overview ----------
with tab_overview:
    health = (
        latest.groupby("asset_id")["status"]
        .apply(lambda s: (s == "ok").mean() * 100)
        .reset_index(name="health_pct")
    )
    overall = (latest["status"] == "ok").mean() * 100 if len(latest) else 0
    failing = int((latest["status"] != "ok").sum()) if len(latest) else 0
    last_status = pipeline.iloc[0]["status"] if len(pipeline) else "no runs yet"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall health", f"{overall:.0f}%")
    c2.metric("Assets tracked", len(assets))
    c3.metric("Failing checks", failing)
    c4.metric("Last pipeline step", last_status)

    st.subheader("Lineage (green = healthy, red = failing, grey = no checks)")
    health_map = dict(zip(health["asset_id"], health["health_pct"]))
    dot = 'digraph { rankdir=LR; node [shape=box, style=filled, fontname=Helvetica];'
    for asset_id in assets["asset_id"]:
        h = health_map.get(asset_id)
        color = "#d9d9d9" if h is None else ("#9be29b" if h == 100 else "#f28b82")
        dot += f' "{asset_id}" [fillcolor="{color}"];'
    for _, e in edges.iterrows():
        dot += f' "{e.upstream_id}" -> "{e.downstream_id}";'
    dot += " }"
    st.graphviz_chart(dot)

    st.subheader("Health by asset")
    st.dataframe(health, use_container_width=True)

# ---------- Data Quality ----------
with tab_quality:
    st.subheader("Latest check results")
    st.dataframe(latest, use_container_width=True)

    st.subheader("History of one check")
    options = latest[["asset_id", "metric"]].drop_duplicates()
    labels = [f"{a} | {m}" for a, m in zip(options.asset_id, options.metric)]
    if labels:
        choice = st.selectbox("Choose a check", labels)
        a, m = [x.strip() for x in choice.split("|")]
        hist = query(
            f"SELECT ts, value FROM events WHERE module = 'data_quality' "
            f"AND asset_id = '{a}' AND metric = '{m}' ORDER BY ts"
        )
        st.line_chart(hist.set_index("ts")["value"])
        st.caption("The chart gets more useful after several pipeline runs.")

# ---------- Pipeline Monitoring ----------
with tab_pipeline:
    if len(pipeline) == 0:
        st.info("No pipeline runs recorded yet. Materialize the assets in Dagster first.")
    else:
        st.subheader("Recent steps")
        st.dataframe(pipeline, use_container_width=True)
        st.subheader("Runtime per step (seconds)")
        st.bar_chart(pipeline.groupby("asset_id")["value"].mean())

# ---------- Real-Time Analytics ----------
with tab_stream:
    s = query("SELECT ts, metric, value, status FROM events "
              "WHERE module = 'streaming' ORDER BY ts")
    if len(s) == 0:
        st.info("No streaming data yet. Run: python -m collectors.streaming")
    else:
        latest_s = s.sort_values("ts").groupby("metric").tail(1)
        cols = st.columns(len(latest_s))
        for col, (_, r) in zip(cols, latest_s.iterrows()):
            icon = "OK" if r["status"] == "ok" else "FAIL"
            col.metric(r["metric"], f"{r['value']:.3f}", icon)
        wide = s.pivot_table(index="ts", columns="metric", values="value")
        st.subheader("Event delay over time (seconds)")
        st.line_chart(wide[["avg_lag_seconds", "p95_lag_seconds"]])
        st.subheader("Late event rate over time")
        st.line_chart(wide[["late_event_rate"]])
        st.caption("Simulated event stream. A spike means events are arriving late.")

# ---------- Cost Intelligence ----------
with tab_cost:
    c = query("SELECT ts, job_name, asset_id, runtime_seconds, cost_usd "
              "FROM costs ORDER BY ts DESC")
    if len(c) == 0:
        st.info("No cost data yet. Run: python -m collectors.cost")
    else:
        by_job = c.groupby("job_name")["cost_usd"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total simulated cost", f"${c['cost_usd'].sum():.2f}")
        c2.metric("Most expensive job", by_job.idxmax())
        c3.metric("Cost records", len(c))
        st.subheader("Cost by job (USD)")
        st.bar_chart(by_job)
        st.subheader("Cost over time (USD per hour)")
        st.line_chart(c.set_index("ts").sort_index().resample("1h")["cost_usd"].sum())
        st.dataframe(c, use_container_width=True)
        st.caption("Simulated price per compute-second. Runtimes are real.")

# ---------- Overview: anomalies ----------
with tab_overview:
    st.subheader("Detected anomalies")
    if len(anoms) == 0:
        st.success("No anomalies detected.")
    else:
        st.warning(f"{len(anoms)} anomalies detected")
        st.dataframe(anoms, use_container_width=True)
        st.caption("A value is flagged when it is more than 3 standard deviations "
                   "away from the average of the previous 10 values.")

# ---------- ML Monitoring ----------
with tab_ml:
    m = query("SELECT ts, metric, value, status FROM events "
              "WHERE module = 'ml' ORDER BY ts")
    if len(m) == 0:
        st.info("No ML data yet. Run: python -m collectors.ml_monitor")
    else:
        latest_m = m.sort_values("ts").groupby("metric").tail(1)
        cols = st.columns(len(latest_m))
        for col, (_, r) in zip(cols, latest_m.iterrows()):
            col.metric(r["metric"], f"{r['value']:.3f}",
                       "OK" if r["status"] == "ok" else "FAIL")
        wide = m.pivot_table(index="ts", columns="metric", values="value")
        st.subheader("Model error over time (MAE, dollars)")
        st.line_chart(wide[["mae"]])
        st.subheader("Input drift over time (PSI of trip distance)")
        st.line_chart(wide[["psi_trip_distance"]])
        st.caption("PSI below 0.1 is stable, above 0.25 is a big shift. "
                   "Each run simulates one new day of taxi data.")


# ---------- AI Reliability ----------
with tab_ai:
    a = query("SELECT ts, metric, value, status, details FROM events "
              "WHERE module = 'ai' ORDER BY ts")
    if len(a) == 0:
        st.info("No AI data yet. Run: python -m collectors.ai_reliability")
    else:
        latest_a = a.sort_values("ts").groupby("metric").tail(1)
        cols = st.columns(len(latest_a))
        for col, (_, r) in zip(cols, latest_a.iterrows()):
            col.metric(r["metric"], f"{r['value']:.3f}",
                       "OK" if r["status"] == "ok" else "FAIL")
        wide = a.pivot_table(index="ts", columns="metric", values="value")
        st.subheader("Answer problems over time")
        st.line_chart(wide[["stale_answer_rate", "unauthorized_answer_rate",
                            "ungrounded_answer_rate"]])
        try:
            bad = query("SELECT ts, mode, role, question, doc_id, answer, stale, "
                        "unauthorized, ungrounded FROM rag_answers "
                        "WHERE stale OR unauthorized OR ungrounded "
                        "ORDER BY ts DESC LIMIT 50")
            st.subheader("Flagged answers")
            st.dataframe(bad, use_container_width=True)
        except Exception:
            pass
        st.caption("Stale = old document version. Unauthorized = the user's role is not "
                   "allowed to see the source. Ungrounded = the answer is not supported "
                   "by any retrieved document.")

# ---------- Governance & Security ----------
with tab_gov:
    g = query("SELECT ts, metric, value, status FROM events "
              "WHERE module = 'governance' ORDER BY ts")
    if len(g) == 0:
        st.info("No governance data yet. Run: python -m collectors.governance")
    else:
        latest_g = g.sort_values("ts").groupby("metric").tail(1)
        cols = st.columns(len(latest_g))
        for col, (_, r) in zip(cols, latest_g.iterrows()):
            col.metric(r["metric"], f"{r['value']:.3f}",
                       "OK" if r["status"] == "ok" else "FAIL")
        wide = g.pivot_table(index="ts", columns="metric", values="value")
        st.subheader("Access violations over time")
        st.line_chart(wide[["access_violation_rate"]])
        try:
            st.subheader("Recent access violations")
            st.dataframe(query("SELECT ts, user_name, role, asset_id FROM access_log "
                               "WHERE NOT allowed ORDER BY ts DESC LIMIT 50"),
                         use_container_width=True)
            pii = query("SELECT * FROM pii_findings")
            st.subheader("Personal data found by the scanner")
            st.dataframe(pii, use_container_width=True)

            import networkx as nx
            graph = nx.DiGraph()
            graph.add_edges_from(zip(edges.upstream_id, edges.downstream_id))
            st.subheader("Where sensitive data flows (lineage)")
            for asset in pii["asset_id"].unique():
                down = sorted(nx.descendants(graph, asset)) if asset in graph else []
                st.write(f"**{asset}** feeds: {', '.join(down) if down else 'nothing downstream'}")
        except Exception as e:
            st.info(f"Waiting for data: {e}")
        st.caption("Simulated access log and fake customer data. The scanner uses "
                   "simple patterns for emails, phone numbers and column names.")

# ---------- Incident Management ----------
with tab_inc:
    inc = query("SELECT incident_id, opened_at, title, root_cause_asset, severity, "
                "status, details FROM incidents ORDER BY severity DESC")
    if len(inc) == 0:
        st.info("No incidents yet. Run: python -m engine.incidents")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Incidents", len(inc))
        c2.metric("Raw alerts", len(anoms))
        c3.metric("Alert reduction", f"{len(anoms) / len(inc):.1f}x")
        st.dataframe(inc.drop(columns=["details"]), use_container_width=True)

        pick = st.selectbox(
            "Investigate an incident", inc["incident_id"].tolist(),
            format_func=lambda i: f"#{i}  " + inc.loc[inc.incident_id == i, "title"].iloc[0])
        d = json.loads(inc.loc[inc.incident_id == pick, "details"].iloc[0])
        root = inc.loc[inc.incident_id == pick, "root_cause_asset"].iloc[0]

        st.subheader("Why this root cause? (score = product of the four parts)")
        st.dataframe(pd.DataFrame(d["ranking"]), use_container_width=True)

        st.subheader("Blast radius (red = root cause, orange = alerting, yellow = affected)")
        dot = 'digraph { rankdir=LR; node [shape=box, style=filled, fontname=Helvetica];'
        for a in assets["asset_id"]:
            color = ("#f28b82" if a == root else
                     "#fbbc04" if a in d["assets_with_alerts"] else
                     "#fff2a8" if a in d["affected"] else "#e8e8e8")
            dot += f' "{a}" [fillcolor="{color}"];'
        for _, e in edges.iterrows():
            dot += f' "{e.upstream_id}" -> "{e.downstream_id}";'
        st.graphviz_chart(dot + " }")

    try:
        res = query("SELECT experiment, method, metric, value, n FROM experiment_results "
                    "ORDER BY experiment, metric, method")
        st.subheader("Experiment results")
        st.dataframe(res, use_container_width=True)
    except Exception:
        st.caption("Experiment results appear here after: python -m experiments.run_experiments")


# ---------- AI Engineering Assistant ----------
with tab_assist:
    st.subheader("AI Engineering Assistant")
    key, model = os.environ.get("GROQ_API_KEY"), os.environ.get("GROQ_MODEL")
    try:
        key = st.secrets.get("GROQ_API_KEY", key)
        model = st.secrets.get("GROQ_MODEL", model)
    except Exception:
        pass
    if "chat" not in st.session_state:
        st.session_state.chat, st.session_state.llm_calls = [], 0
    use_key = key if st.session_state.llm_calls < 15 else None   # cap per visitor
    st.caption("Mode: " + ("language model + read-only tools" if use_key
                           else "built-in rules (no API key, or session limit reached)")
               + ". Every answer comes from the metadata store.")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    prompt = st.chat_input("Try: Why did incident 1 happen?  /  What should we do about it?")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.spinner("Investigating..."):
            try:
                text, used = assistant.answer(prompt, st.session_state.chat[:-1], use_key, model)
            except Exception as e:
                text, used = f"Could not answer: {e}", "rules"
        st.session_state.llm_calls += used == "llm"
        st.session_state.chat.append({"role": "assistant", "content": text})
        st.rerun()

# ---------- Experiments ----------
with tab_exp:
    try:
        res = query("SELECT experiment, method, metric, value, n FROM experiment_results "
                    "ORDER BY experiment, method, metric")
    except Exception:
        res = pd.DataFrame()
    if res.empty:
        st.info("No experiment results yet. Run the scripts in experiments/.")
    for name, g in res.groupby("experiment"):
        st.subheader(name.replace("_", " ").title())
        st.caption(f"{int(g['n'].iloc[0])} test cases")
        st.dataframe(g.pivot_table(index="method", columns="metric", values="value").round(3),
                     use_container_width=True)