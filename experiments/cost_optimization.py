import networkx as nx
import numpy as np

from experiments.common import save_results
from experiments.lab import Lab

LEAVES = ["exec_dashboard", "ops_dashboard", "ml_features", "avg_fare_by_pax"]
MIDS = ["daily_revenue", "borough_revenue", "hourly_demand", "fare_features",
        "trips_enriched", "kpi_summary"]
LINEAGE_CONSUMERS = ["exec_dashboard", "ops_dashboard", "ml_features"]
CANDIDATES = [24, 12, 8, 6, 4, 3, 2]     # refresh every N hours, tried largest first
N_SCENARIOS = 40


def simulate(lab, schedule, reads):
    """Count reads that meet their freshness limit. Sources are ingested every hour;
    a derived table refreshed at hour h gets the oldest data time of its parents."""
    data = {t: -1 for t in lab.order}
    ok = 0
    for h in range(24):
        for s in lab.sources:
            data[s] = h
        for t in lab.order:
            if t not in lab.sources and h in schedule.get(t, ()):
                data[t] = min(data[p] for p in lab.parents[t])
        for table, hour, sla in reads:
            if hour == h:
                ok += (h - data[table]) <= sla
    return int(ok)


def needed(lab, reads):
    tables = {t for t, _, _ in reads}
    out = set(tables)
    for t in tables:
        out |= nx.ancestors(lab.graph, t)
    return {t for t in out if t not in lab.sources}


def make_scenario(rng):
    reads = []
    for leaf in LEAVES:
        if rng.random() < 0.65:
            sla = int(rng.choice([1, 2, 4, 8]))
            for h in rng.choice(np.arange(6, 23), size=int(rng.integers(1, 5)), replace=False):
                reads.append((leaf, int(h), sla))
    if not reads:
        reads.append(("exec_dashboard", 12, 4))
    if rng.random() < 0.5:                       # shadow consumer on a middle table
        t, sla = str(rng.choice(MIDS)), int(rng.choice([2, 4, 8]))
        for h in rng.choice(np.arange(8, 20), size=2, replace=False):
            reads.append((t, int(h), sla))
    hidden = []
    if rng.random() < 0.25:                      # a read the usage log never saw
        hidden.append((str(rng.choice(LEAVES + MIDS)), int(rng.integers(6, 23)), 4))
    return reads, hidden


def build_schedules(lab, derived, cost, observed):
    hourly = lambda names: {t: (set(range(24)) if t in names else set()) for t in derived}
    use = needed(lab, observed)
    ours = hourly(use)
    for t in sorted(use, key=lambda x: -cost[x]):
        for p in CANDIDATES:
            trial = dict(ours)
            trial[t] = set(range(0, 24, p))
            if simulate(lab, trial, observed) == len(observed):   # what-if check
                ours = trial
                break
    return {
        "original (everything hourly)": hourly(set(derived)),
        "baseline: blind cut to every 6h": {t: set(range(0, 24, 6)) for t in derived},
        "baseline: lineage-only cleanup": hourly(needed(lab, [(c, 0, 0) for c in LINEAGE_CONSUMERS])),
        "baseline: usage-log cleanup, no tuning": hourly(use),
        "ours: usage log + SLA what-if": ours,
    }


def main():
    lab = Lab()
    derived = [t for t in lab.order if t not in lab.sources]
    tables = lab.build(1)
    cost = {t: sum(len(tables[p]) for p in lab.parents[t]) / 1000 + 0.5 for t in derived}
    full_cost = sum(24 * cost[t] for t in derived)

    stats = {}
    for i in range(N_SCENARIOS):
        observed, hidden = make_scenario(np.random.default_rng(i))
        true_reads = observed + hidden
        truth_unneeded = set(derived) - needed(lab, true_reads)
        for name, sched in build_schedules(lab, derived, cost, observed).items():
            s = stats.setdefault(name, dict(cost=[], comp=[], safe=[], tp=0, fp=0, fn=0))
            s["cost"].append(sum(len(sched[t]) * cost[t] for t in derived) / full_cost)
            comp = simulate(lab, sched, true_reads) / len(true_reads)
            s["comp"].append(comp)
            s["safe"].append(comp == 1.0)
            removed = {t for t in derived if not sched[t]}
            s["tp"] += len(removed & truth_unneeded)
            s["fp"] += len(removed - truth_unneeded)
            s["fn"] += len(truth_unneeded - removed)

    results = []
    print(f"{N_SCENARIOS} random company setups\n")
    for name, s in stats.items():
        mean = lambda v: sum(v) / len(v)
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 1.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 1.0
        results += [(name, "cost_vs_original", mean(s["cost"])),
                    (name, "sla_compliance", mean(s["comp"])),
                    (name, "fully_safe_setup_rate", mean(s["safe"])),
                    (name, "removal_precision", prec), (name, "removal_recall", rec)]
        print(f"{name:40} cost {mean(s['cost']):.2f}  SLA met {mean(s['comp']):.3f}  "
              f"fully safe {mean(s['safe']):.2f}  removal precision {prec:.2f} recall {rec:.2f}")
    save_results("cost_optimization", results, N_SCENARIOS)
    print("\nSaved to the metadata store.")


if __name__ == "__main__":
    main()