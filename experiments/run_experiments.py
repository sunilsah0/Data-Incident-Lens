import os
from datetime import datetime

import networkx as nx
import pandas as pd

from engine.incidents import group_alerts, rank_root_causes
from engine.store import get_store, init_store
from experiments.lab import (ALL_FAULTS, Lab, baseline_stats, detect, fingerprints,
                             predict_affected, table_metrics)

SEEDS_PER_FAULT = 6
BASELINE_SEEDS = list(range(1000, 1015))   # 15 clean "days" to learn normal behaviour


def precision_recall(pred, truth):
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 1.0
    return p, r


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def main():
    lab = Lab()
    print("Learning normal behaviour from 15 clean runs...")
    stats = baseline_stats(lab, BASELINE_SEEDS)
    rows = []
    total = len(ALL_FAULTS) * SEEDS_PER_FAULT
    done = 0
    for fault in ALL_FAULTS:
        for k in range(SEEDS_PER_FAULT):
            seed = 5000 + k
            clean, faulty = lab.build(seed), lab.build(seed, fault)
            fp_c, fp_f = fingerprints(clean), fingerprints(faulty)
            truth = {n for n in fp_c if fp_c[n] != fp_f[n]}
            done += 1
            if fault["node"] not in truth:
                rows.append(dict(fault=fault["id"], seed=seed, has_effect=False))
                continue

            alerts = detect(table_metrics(faulty), stats)
            strengths = {}
            for node, _, z in alerts:
                strengths[node] = max(strengths.get(node, 0.0), min(abs(z), 10) / 10)

            row = dict(fault=fault["id"], kind=fault["kind"], seed=seed, has_effect=True,
                       true_root=fault["node"], truth_size=len(truth),
                       alerts=len(alerts), detected=bool(strengths),
                       ours_top1=False, ours_top3=False, loud_top1=False, upstream_top1=False,
                       incidents=0)
            if strengths:
                ranked = rank_root_causes(lab.graph, strengths)
                names = [r["asset"] for r in ranked]
                row["ours_top1"] = names[0] == fault["node"]
                row["ours_top3"] = fault["node"] in names[:3]
                row["loud_top1"] = max(strengths, key=strengths.get) == fault["node"]
                row["upstream_top1"] = min(strengths, key=lab.order.index) == fault["node"]
                row["incidents"] = len(group_alerts(lab.graph, strengths.keys()))

            table_pred = {fault["node"]} | nx.descendants(lab.graph, fault["node"])
            col_pred = predict_affected(lab, fault)
            row["table_p"], row["table_r"] = precision_recall(table_pred, truth)
            row["col_p"], row["col_r"] = precision_recall(col_pred, truth)
            rows.append(row)
        print(f"  {done}/{total} runs done ({fault['id']})")

    runs = pd.DataFrame(rows)
    os.makedirs("experiments/results", exist_ok=True)
    runs.to_csv("experiments/results/lab_runs.csv", index=False)
    eff = runs[runs["has_effect"]].copy()
    det = eff[eff["detected"]]

    results = []   # (experiment, method, metric, value)
    for method, col in [("ours: explainable score", "ours_top1"),
                        ("baseline: loudest alert", "loud_top1"),
                        ("baseline: most upstream alert", "upstream_top1")]:
        results.append(("root_cause", method, "top1_accuracy", eff[col].mean()))
    results.append(("root_cause", "ours: explainable score", "top3_accuracy", eff["ours_top3"].mean()))
    results.append(("root_cause", "all methods", "fault_detection_rate", eff["detected"].mean()))
    results.append(("alert_noise", "raw alerts per fault", "mean", det["alerts"].mean()))
    results.append(("alert_noise", "incidents per fault", "mean", det["incidents"].mean()))
    results.append(("alert_noise", "reduction factor", "x", det["alerts"].sum() / max(det["incidents"].sum(), 1)))
    for name, p, r in [("table-level lineage", "table_p", "table_r"),
                       ("column-level lineage", "col_p", "col_r")]:
        results.append(("dependency", name, "precision", eff[p].mean()))
        results.append(("dependency", name, "recall", eff[r].mean()))
        results.append(("dependency", name, "f1",
                        float(pd.Series([f1(a, b) for a, b in zip(eff[p], eff[r])]).mean())))

    print(f"\nFaults with a real effect: {len(eff)} of {len(runs)} runs")
    for exp_name, method, metric, value in results:
        print(f"{exp_name:12} {method:32} {metric:22} {value:8.3f}")
    print("\nTop-1 accuracy per fault type (ours / loudest / most upstream):")
    print(eff.groupby("fault")[["detected", "ours_top1", "loud_top1", "upstream_top1"]]
          .mean().round(2).to_string())

    init_store()
    con = get_store()
    con.execute("CREATE TABLE IF NOT EXISTS experiment_results (experiment VARCHAR, "
                "method VARCHAR, metric VARCHAR, value DOUBLE, n INTEGER, ts TIMESTAMP)")
    con.execute("DELETE FROM experiment_results WHERE experiment IN "
                "('root_cause', 'alert_noise', 'dependency')")
    now = datetime.now()
    for exp_name, method, metric, value in results:
        con.execute("INSERT INTO experiment_results VALUES (?,?,?,?,?,?)",
                    [exp_name, method, metric, float(value), len(eff), now])
    con.close()
    print("\nSaved to the metadata store and experiments/results/lab_runs.csv")


if __name__ == "__main__":
    main()