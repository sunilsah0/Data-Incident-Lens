import json

import networkx as nx
import pandas as pd
from engine.store import get_store, init_store

TIME_GAP_MINUTES = 10   # alerts further apart than this are separate events


def group_alerts(graph, nodes):
    """Merge anomalous assets that are connected by lineage into one incident."""
    nodes = list(nodes)
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u in nodes:
        if u not in graph:
            continue
        below = nx.descendants(graph, u)
        for v in nodes:
            if v in below:
                parent[find(u)] = find(v)
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


def rank_root_causes(graph, strengths):
    """strengths: {asset: 0..1, how abnormal}. Returns candidates, best first."""
    anomalous = set(strengths)
    total = len(anomalous)
    ranked = []
    for c in anomalous:
        below = nx.descendants(graph, c) if c in graph else set()
        above = nx.ancestors(graph, c) if c in graph else set()
        region = below | {c}
        hit = anomalous & region
        own = float(strengths[c])
        explained = len(hit) / total                 # share of all alerts explained
        purity = len(hit) / len(region)              # how much of the region is truly hit
        clean = 1.0 / (1 + len(anomalous & above))   # broken ancestors: probably a victim
        ranked.append({
            "asset": c,
            "score": round(own * explained * purity * clean, 4),
            "own": round(own, 3),
            "explained": round(explained, 3),
            "purity": round(purity, 3),
            "clean_above": round(clean, 3),
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


def blast_radius(graph, root):
    return {root} | (nx.descendants(graph, root) if root in graph else set())

def peel_roots(graph, strengths, max_roots=3):
    """Pick the best root cause, remove the alerts it explains, repeat."""
    remaining = dict(strengths)
    roots = []
    while remaining and len(roots) < max_roots:
        root = rank_root_causes(graph, remaining)[0]["asset"]
        roots.append(root)
        region = blast_radius(graph, root)
        remaining = {a: s for a, s in remaining.items() if a not in region}
    return roots

def run_live():
    init_store()
    con = get_store()
    edges = con.execute("SELECT upstream_id, downstream_id FROM lineage_edges").fetchall()
    crit = dict(con.execute("SELECT asset_id, criticality FROM assets").fetchall())
    alerts = con.execute(
        "SELECT ts, asset_id, module, metric, zscore FROM anomalies ORDER BY ts").df()
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    con.execute("DELETE FROM incidents")
    if alerts.empty:
        con.close()
        print("No anomalies, so no incidents.")
        return

    alerts["cluster"] = (alerts["ts"].diff() > pd.Timedelta(minutes=TIME_GAP_MINUTES)).cumsum()
    next_id = 1
    for _, batch in alerts.groupby("cluster"):
        strengths = {a: float(min(g["zscore"].abs().max(), 10) / 10)
                     for a, g in batch.groupby("asset_id")}
        for group in group_alerts(graph, strengths.keys()):
            sub = {a: strengths[a] for a in group}
            ranked = rank_root_causes(graph, sub)
            root = ranked[0]["asset"]
            region = blast_radius(graph, root)
            severity = float(round(sum(crit.get(a, 1) for a in region)
                                   * (0.5 + 0.5 * max(sub.values())), 2))
            mine = batch[batch["asset_id"].isin(group)]
            title = f"{len(mine)} alerts on {len(group)} asset(s), likely root cause: {root}"
            details = json.dumps({
                "ranking": ranked[:5],
                "affected": sorted(region),
                "assets_with_alerts": sorted(group),
                "metrics": sorted(set(mine["metric"])),
            }, default=str)
            con.execute("INSERT INTO incidents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [next_id, mine["ts"].min(), title, root, severity, 0.0,
                         "open", details])
            next_id += 1
    con.close()
    n_alerts, n_inc = len(alerts), next_id - 1
    print(f"{n_alerts} alerts became {n_inc} incidents")


if __name__ == "__main__":
    run_live()