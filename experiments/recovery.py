import duckdb
import networkx as nx
import numpy as np

from engine.incidents import rank_root_causes
from experiments.common import blast_from_root, save_results
from experiments.lab import (DATA_FAULTS, SAMPLE_ROWS, Lab, baseline_stats, detect,
                             fingerprints, table_metrics)

N_DAYS = 14
DAY_SEEDS = [9000 + i for i in range(N_DAYS)]
BASELINE_SEEDS = list(range(1000, 1015))
REPEATS = 2


def rebuild(lab, seed, names, state):
    """Re-ingest clean sources and re-run the SQL for the tables in `names`.
    All other tables are kept as they are."""
    trips = lab.pool.sample(SAMPLE_ROWS, random_state=seed).reset_index(drop=True)
    zones = lab.zones.copy()
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=1")
    for name in lab.order:
        if name in names and name in lab.sources:
            con.register(f"{name}_src", trips if name == "trips_raw" else zones)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_src")
        elif name in names:
            con.execute(f"CREATE TABLE {name} AS {lab.sql[name]}")
        else:
            con.register(f"{name}_src", state[name])
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_src")
        if name in names:
            state[name] = con.execute(f"SELECT * FROM {name}").df()
    con.close()


def scan(lab, states, stats):
    """Run monitoring on every day. Returns damaged days and alert strengths."""
    flagged, strengths = [], {}
    for d, state in states.items():
        alerts = detect(table_metrics(state), stats)
        if alerts:
            flagged.append(d)
            for node, _, z in alerts:
                strengths[node] = max(strengths.get(node, 0.0), min(abs(z), 10) / 10)
    return flagged, strengths


def plan(lab, level, root, flagged, all_days):
    if level == 0:      # targeted: tables in the column-level blast radius
        return blast_from_root(lab, root), flagged
    if level == 1:      # targeted: everything downstream at table level
        return {root} | nx.descendants(lab.graph, root), flagged
    if level == 2:      # all tables, damaged days only
        return set(lab.order), flagged
    return set(lab.order), all_days    # full replay


def run_planner(lab, states, stats, all_days):
    flagged, strengths = scan(lab, states, stats)
    if not flagged:
        return dict(detected=False, cost=0, level=None, validated=False)
    root = rank_root_causes(lab.graph, strengths)[0]["asset"]
    cost = 0
    for level in range(4):
        names, days = plan(lab, level, root, flagged, all_days)
        for d in days:
            rebuild(lab, d, names, states[d])
        cost += len(names) * len(days)
        left, _ = scan(lab, states, stats)        # validation with the monitors
        if not left:
            return dict(detected=True, cost=cost, level=level, validated=True)
    return dict(detected=True, cost=cost, level=3, validated=False)


def is_correct(states, golden_fp):
    return all(fingerprints(states[d]) == golden_fp[d] for d in states)


def main():
    lab = Lab()
    stats = baseline_stats(lab, BASELINE_SEEDS)
    full_cost = len(lab.order) * N_DAYS
    print("Building 14 clean days...")
    golden = {s: lab.build(s) for s in DAY_SEEDS}
    golden_fp = {s: fingerprints(golden[s]) for s in DAY_SEEDS}

    rows = []
    for fi, fault in enumerate(DATA_FAULTS):
        for rep in range(REPEATS):
            rng = np.random.default_rng(fi * 10 + rep)
            if fault["node"] == "zones_raw":
                bad_days = list(DAY_SEEDS)                    # a wrong lookup table hits every day
            else:
                start = int(rng.integers(0, N_DAYS - 2))
                bad_days = DAY_SEEDS[start:start + 3]         # 3 bad days in a row
            states = {s: dict(golden[s]) for s in DAY_SEEDS}
            for s in bad_days:
                states[s] = lab.build(s, fault)
            if is_correct(states, golden_fp):
                continue                                      # fault had no effect

            planner_states = {s: dict(v) for s, v in states.items()}
            res = run_planner(lab, planner_states, stats, DAY_SEEDS)
            planner_ok = is_correct(planner_states, golden_fp)

            part_states = {s: dict(v) for s, v in states.items()}
            flagged, _ = scan(lab, part_states, stats)
            for d in flagged:
                rebuild(lab, d, set(lab.order), part_states[d])
            part_cost = len(lab.order) * len(flagged)
            part_ok = is_correct(part_states, golden_fp)

            rows.append(dict(fault=fault["id"], bad=len(bad_days), flagged=len(flagged),
                             level=res["level"], cost=res["cost"], ok=planner_ok,
                             validated=res["validated"], detected=res["detected"],
                             part_cost=part_cost, part_ok=part_ok))
            print(f"{fault['id']:16} rep{rep}  bad days {len(bad_days):2}  flagged {len(flagged):2}  "
                  f"planner level {res['level']}  cost {res['cost']:4}/{full_cost}  "
                  f"correct {planner_ok}")

    n = len(rows)
    mean = lambda key: sum(r[key] for r in rows) / n
    results = [
        ("full replay", "cost_vs_full_replay", 1.0),
        ("full replay", "correct_rate", 1.0),
        ("partition backfill only", "cost_vs_full_replay", mean("part_cost") / full_cost),
        ("partition backfill only", "correct_rate", mean("part_ok")),
        ("do nothing (fix forward)", "cost_vs_full_replay", 0.0),
        ("do nothing (fix forward)", "correct_rate", 0.0),
        ("ours: plan, validate, escalate", "cost_vs_full_replay", mean("cost") / full_cost),
        ("ours: plan, validate, escalate", "correct_rate", mean("ok")),
        ("ours: plan, validate, escalate", "first_try_rate",
         sum(r["level"] == 0 for r in rows) / n),
        ("ours: plan, validate, escalate", "silent_residual_damage_rate",
         sum(r["validated"] and not r["ok"] for r in rows) / n),
    ]
    print(f"\n{n} recovery scenarios (full replay costs {full_cost} table rebuilds)")
    for method, metric, value in results:
        print(f"{method:34} {metric:30} {value:7.3f}")
    save_results("recovery", results, n)
    print("\nSaved to the metadata store.")


if __name__ == "__main__":
    main()