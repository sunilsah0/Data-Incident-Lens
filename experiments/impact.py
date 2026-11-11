import networkx as nx
import numpy as np
import pandas as pd

from engine.incidents import rank_root_causes
from experiments.common import VALUE_PER_DAY, blast_from_root, save_results
from experiments.lab import (ALL_FAULTS, Lab, baseline_stats, detect, fingerprints,
                             table_metrics)

SEEDS = [6000, 6001]
BASELINE_SEEDS = list(range(1000, 1015))


def damage(clean, faulty):
    """0..1: how badly did this table change? (rows, averages, null rates)"""
    d = abs(len(faulty) - len(clean)) / len(clean) if len(clean) else 0.0
    for col in clean.columns:
        if col not in faulty.columns:
            continue
        d = max(d, float(faulty[col].isna().mean() - clean[col].isna().mean()))
        numeric = pd.api.types.is_numeric_dtype(clean[col]) and not pd.api.types.is_bool_dtype(clean[col])
        if numeric and col != "trip_id":
            a, b = clean[col].mean(), faulty[col].mean()
            if pd.notna(a) and pd.notna(b) and abs(a) > 1e-9:
                d = max(d, abs(b - a) / abs(a))
            elif pd.notna(a) and pd.isna(b):
                d = max(d, 1.0)
    return min(1.0, d)


def top_overlap(est, true, q=0.25):
    k = max(1, int(round(len(true) * q)))
    top_true = set(np.argsort(-np.array(true), kind="stable")[:k])
    top_est = set(np.argsort(-np.array(est), kind="stable")[:k])
    return len(top_true & top_est) / k


def main():
    lab = Lab()
    stats = baseline_stats(lab, BASELINE_SEEDS)
    true_sev, ours, downstream, alert_count = [], [], [], []
    for fault in ALL_FAULTS:
        for seed in SEEDS:
            clean, faulty = lab.build(seed), lab.build(seed, fault)
            fc, ff = fingerprints(clean), fingerprints(faulty)
            if all(fc[n] == ff[n] for n in fc):
                continue                       # the fault had no effect
            total = 0.0
            for n in clean:
                if fc[n] != ff[n]:
                    total += VALUE_PER_DAY[n] * max(damage(clean[n], faulty[n]), 0.02)
            alerts = detect(table_metrics(faulty), stats)
            strengths = {}
            for node, _, z in alerts:
                strengths[node] = max(strengths.get(node, 0.0), min(abs(z), 10) / 10)
            if strengths:
                root = rank_root_causes(lab.graph, strengths)[0]["asset"]
                region = blast_from_root(lab, root)
                ours.append(strengths[root] * sum(lab.crit[t] for t in region))
                downstream.append(1 + len(nx.descendants(lab.graph, root)))
                alert_count.append(len(alerts))
            else:                              # fault not detected: all methods see nothing
                ours.append(0.0)
                downstream.append(0)
                alert_count.append(0)
            true_sev.append(total)

    truth = pd.Series(true_sev)
    results = []
    print(f"{len(truth)} faults with a real effect\n")
    for label, est in [("ours: alert strength x weighted blast radius", ours),
                       ("baseline: number of downstream tables", downstream),
                       ("baseline: number of alerts", alert_count)]:
        rho = pd.Series(est).corr(truth, method="spearman")
        rho = 0.0 if pd.isna(rho) else float(rho)
        top = top_overlap(est, true_sev)
        results += [(label, "spearman_rank_correlation", rho),
                    (label, "top25pct_overlap", top)]
        print(f"{label:48} spearman {rho:6.3f}   top-25% overlap {top:.2f}")
    save_results("business_impact", results, len(truth))
    print("\nSaved to the metadata store.")


if __name__ == "__main__":
    main()