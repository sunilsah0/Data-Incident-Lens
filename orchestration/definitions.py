import os
import sys
import time
import runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["DUCKDB_PATH"] = os.path.join(ROOT, "data", "incidentlens.duckdb")

from dagster import (
    asset, Definitions, ScheduleDefinition, define_asset_job, AssetSelection
)
from engine.store import init_store, get_store
from collectors.lineage import collect_lineage
from collectors.quality import collect_quality


def logged(name, fn, raise_on_fail=True):
    """Run a step, time it, and record the result in the events table."""
    start = time.time()
    try:
        fn()
        status, details = "ok", ""
    except Exception as e:
        status, details = "fail", str(e)[:500]
    seconds = time.time() - start
    init_store()
    con = get_store()
    con.execute(
        "INSERT INTO events (asset_id, module, metric, value, status, details) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [f"pipeline.{name}", "pipeline", "runtime_seconds", seconds, status, details],
    )
    con.close()
    if status == "fail" and raise_on_fail:
        raise RuntimeError(f"{name} failed: {details}")


def _dbt(command):
    from dbt.cli.main import dbtRunner
    res = dbtRunner().invoke(
        [command, "--project-dir", "pipelines", "--profiles-dir", "pipelines"]
    )
    if not res.success:
        msg = str(res.exception) if res.exception else ""
        results = getattr(res.result, "results", None) or []
        bad = [f"{r.node.name}: {r.message}" for r in results
               if r.status in ("error", "fail")]
        raise RuntimeError(f"dbt {command} failed: {msg} {' | '.join(bad)}"[:500])

@asset
def raw_data():
    logged("load_raw", lambda: runpy.run_path("collectors/load_raw.py"))


@asset(deps=[raw_data])
def dbt_models():
    logged("dbt_run", lambda: _dbt("run"))


@asset(deps=[dbt_models])
def dbt_tests():
    # A failing test is recorded as an event but does not stop the pipeline,
    # because we still want the quality checks to run and report.
    logged("dbt_test", lambda: _dbt("test"), raise_on_fail=False)


# Lineage and quality both write to the metadata store,
# so they run one after the other, not at the same time.
@asset(deps=[dbt_tests])
def lineage_graph():
    logged("lineage", collect_lineage)


@asset(deps=[lineage_graph])
def quality_checks():
    logged("quality", collect_quality)


daily_job = define_asset_job("daily_pipeline", selection=AssetSelection.all())
daily_schedule = ScheduleDefinition(job=daily_job, cron_schedule="0 6 * * *")

defs = Definitions(
    assets=[raw_data, dbt_models, dbt_tests, lineage_graph, quality_checks],
    jobs=[daily_job],
    schedules=[daily_schedule],
)