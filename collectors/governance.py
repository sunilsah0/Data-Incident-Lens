import random
import re
import sys
from datetime import datetime

import duckdb
from engine.store import get_store, init_store, register_asset, SOURCE_PATH

FIRST = ["Anna", "Erik", "Maria", "Johan", "Sara", "Omar", "Lena", "Ravi", "Mina", "Tom"]
LAST = ["Berg", "Nilsson", "Khan", "Lund", "Sah", "Ali", "Holm", "Rao", "Dahl", "Ek"]
CITIES = ["Stockholm", "Uppsala", "Malmo", "Gothenburg"]

ROLE_LEVEL = {"intern": 0, "contractor": 0, "analyst": 1, "manager": 2}
USERS = {"alice": "analyst", "bob": "analyst", "omar": "analyst",
         "carla": "manager", "dev": "manager", "lena": "manager",
         "ivan": "intern", "joy": "contractor"}
ASSET_LEVEL = {
    "raw.yellow_trips": 1, "main.stg_trips": 1,
    "main.mart_daily_revenue": 0, "main.mart_zone_trips": 0,
    "ml.fare_model": 1, "dash.revenue_dashboard": 0, "dash.zone_report": 0,
    "raw.customers_sim": 2, "dash.customer_report": 2,
}
PII_ASSETS = {"raw.customers_sim", "dash.customer_report"}

# (asset_id, type, name, criticality, upstream asset)
BUSINESS = [
    ("raw.customers_sim", "sim_source", "customers_sim", 3, None),
    ("dash.revenue_dashboard", "dashboard", "revenue_dashboard", 3, "main.mart_daily_revenue"),
    ("dash.zone_report", "dashboard", "zone_report", 2, "main.mart_zone_trips"),
    ("dash.customer_report", "dashboard", "customer_report", 3, "raw.customers_sim"),
    ("ml.fare_model", "ml_model", "fare_model", 3, "main.stg_trips"),
]

NAME_HINTS = ["full_name", "first_name", "last_name", "email", "phone", "ssn", "address"]
EMAIL = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.]+$")
PHONE = re.compile(r"^\+?\d[\d\s-]{7,}$")


def ensure_customers():
    con = duckdb.connect(SOURCE_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'raw' AND table_name = 'customers_sim'").fetchone()[0]
    if not exists:
        rng = random.Random(42)
        rows = []
        for i in range(1, 501):
            f, l = rng.choice(FIRST), rng.choice(LAST)
            rows.append((i, f"{f} {l}", f"{f.lower()}.{l.lower()}{i}@example.com",
                         f"+4670{rng.randint(1000000, 9999999)}", rng.choice(CITIES)))
        con.execute("CREATE TABLE raw.customers_sim (customer_id INTEGER, full_name VARCHAR, "
                    "email VARCHAR, phone VARCHAR, city VARCHAR)")
        con.executemany("INSERT INTO raw.customers_sim VALUES (?,?,?,?,?)", rows)
        print("Created synthetic raw.customers_sim (all data is fake)")
    con.close()


def register_business_assets():
    con = get_store()
    for asset_id, atype, name, crit, upstream in BUSINESS:
        con.execute("INSERT OR REPLACE INTO assets VALUES (?, ?, ?, ?)",
                    [asset_id, atype, name, crit])
        if upstream:
            con.execute("DELETE FROM lineage_edges WHERE upstream_id = ? AND downstream_id = ?",
                        [upstream, asset_id])
            con.execute("INSERT INTO lineage_edges VALUES (?, ?)", [upstream, asset_id])
    con.close()


def scan_pii():
    con = duckdb.connect(SOURCE_PATH, read_only=True)
    cols = con.execute(
        "SELECT table_schema, table_name, column_name FROM information_schema.columns "
        "WHERE table_schema IN ('raw', 'main') AND data_type = 'VARCHAR'").fetchall()
    findings = []
    for schema, table, col in cols:
        values = [str(r[0]) for r in con.execute(
            f'SELECT "{col}" FROM {schema}.{table} WHERE "{col}" IS NOT NULL LIMIT 1000').fetchall()]
        if not values:
            continue
        email_rate = sum(bool(EMAIL.match(v)) for v in values) / len(values)
        phone_rate = sum(bool(PHONE.match(v)) for v in values) / len(values)
        hint = any(h in col.lower() for h in NAME_HINTS)
        if email_rate > 0.5:
            findings.append((f"{schema}.{table}", col, "email", email_rate))
        elif phone_rate > 0.5:
            findings.append((f"{schema}.{table}", col, "phone", phone_rate))
        elif hint:
            findings.append((f"{schema}.{table}", col, "personal (column name)", 1.0))
    con.close()
    return findings


def collect_governance(breach=False):
    init_store()
    ensure_customers()
    register_business_assets()
    findings = scan_pii()

    con = get_store()
    con.execute("CREATE TABLE IF NOT EXISTS pii_findings "
                "(asset_id VARCHAR, column_name VARCHAR, pii_type VARCHAR, match_rate DOUBLE)")
    con.execute("CREATE TABLE IF NOT EXISTS access_log "
                "(ts TIMESTAMP, user_name VARCHAR, role VARCHAR, asset_id VARCHAR, "
                "action VARCHAR, allowed BOOLEAN)")
    con.execute("DELETE FROM pii_findings")
    con.executemany("INSERT INTO pii_findings VALUES (?,?,?,?)", findings)

    rng = random.Random()
    users = list(USERS.items())
    events = []
    for _ in range(400):
        user, role = rng.choice(users)
        allowed_assets = [a for a, lvl in ASSET_LEVEL.items() if lvl <= ROLE_LEVEL[role]]
        # 1% of the time a user touches something random (a normal mistake)
        asset = rng.choice(allowed_assets) if rng.random() > 0.01 else rng.choice(list(ASSET_LEVEL))
        events.append((user, role, asset))
    if breach:
        for _ in range(40):
            user, role = rng.choice([("ivan", "intern"), ("joy", "contractor")])
            events.append((user, role, rng.choice(sorted(PII_ASSETS))))

    now = datetime.now()
    rows = [(now, u, r, a, "read", ROLE_LEVEL[r] >= ASSET_LEVEL[a]) for u, r, a in events]
    con.executemany("INSERT INTO access_log VALUES (?,?,?,?,?,?)", rows)

    violations = [r for r in rows if not r[5]]
    metrics = {
        "access_violation_rate": len(violations) / len(rows),
        "pii_unauthorized_accesses": float(sum(r[3] in PII_ASSETS for r in violations)),
        "pii_columns_found": float(len(findings)),
    }
    print(f"Breach mode: {breach}")
    for name, value in metrics.items():
        bad = (name == "access_violation_rate" and value > 0.05) or \
              (name == "pii_unauthorized_accesses" and value > 5)
        status = "fail" if bad else "ok"
        con.execute(
            "INSERT INTO events (ts, asset_id, module, metric, value, status, details) "
            "VALUES (?, 'governance.access', 'governance', ?, ?, ?, ?)",
            [now, name, value, status, "breach" if breach else "normal"])
        print(f"  {name:28} {value:8.3f}  [{status}]")
    con.close()


if __name__ == "__main__":
    collect_governance(breach="--breach" in sys.argv)