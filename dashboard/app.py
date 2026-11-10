import duckdb
import streamlit as st

STORE = "data/metadata.duckdb"
st.set_page_config(page_title="IncidentLens", layout="wide")


@st.cache_data(ttl=10)
def query(sql):
    con = duckdb.connect(STORE, read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


st.title("IncidentLens")
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
except Exception as e:
    st.error(f"Could not read the metadata store. Is Dagster writing to it right now? Refresh in a few seconds.\n\n{e}")
    st.stop()

tab_overview, tab_quality, tab_pipeline = st.tabs(
    ["Overview", "Data Quality", "Pipeline Monitoring"]
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