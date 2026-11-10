import os
import duckdb

STORE_PATH = "data/metadata.duckdb"   # the control-tower database
SOURCE_PATH = "data/incidentlens.duckdb"  # the taxi data + dbt models


def get_store():
    os.makedirs("data", exist_ok=True)
    return duckdb.connect(STORE_PATH)


def init_store():
    con = get_store()
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_events START 1")
    con.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            asset_id VARCHAR PRIMARY KEY,
            asset_type VARCHAR,
            name VARCHAR,
            criticality INTEGER
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS lineage_edges (
            upstream_id VARCHAR,
            downstream_id VARCHAR
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id BIGINT DEFAULT nextval('seq_events'),
            ts TIMESTAMP DEFAULT current_timestamp,
            asset_id VARCHAR,
            module VARCHAR,
            metric VARCHAR,
            value DOUBLE,
            status VARCHAR,
            details VARCHAR
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS costs (
            ts TIMESTAMP DEFAULT current_timestamp,
            job_name VARCHAR,
            asset_id VARCHAR,
            runtime_seconds DOUBLE,
            cost_usd DOUBLE
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            incident_id BIGINT,
            opened_at TIMESTAMP,
            title VARCHAR,
            root_cause_asset VARCHAR,
            severity DOUBLE,
            cost_usd DOUBLE,
            status VARCHAR,
            details VARCHAR
        )""")
    con.execute("""
    CREATE TABLE IF NOT EXISTS anomalies (
        event_id BIGINT,
        ts TIMESTAMP,
        asset_id VARCHAR,
        module VARCHAR,
        metric VARCHAR,
        value DOUBLE,
        expected DOUBLE,
        zscore DOUBLE
    )""")
    con.close()

def register_asset(asset_id, asset_type, name, criticality):
    con = get_store()
    con.execute("INSERT OR REPLACE INTO assets VALUES (?, ?, ?, ?)",
                [asset_id, asset_type, name, criticality])
    con.close()


if __name__ == "__main__":
    init_store()
    print("Metadata store ready:", STORE_PATH)