import random

from engine.incidents import peel_roots
from experiments.common import save_results
from experiments.lab import ALL_FAULTS, Lab, baseline_stats, detect, table_metrics

N_PAIRS = 40
BASELINE_SEEDS = list(range(1000, 1015))


def main():
    lab = Lab()
    stats = baseline_stats(lab, BASELINE_SEEDS)
    rng = random.Random(7)
    pairs = []
    while len(pairs) < N_PAIRS:
        a, b = rng.sample(ALL_FAULTS, 2)
        if a["node"] != b["node"]:
            pairs.append((a, b))

    scores = {"ours: peel roots": [], "baseline: loudest alert": [],
              "baseline: most upstream alert": []}
    both = {k: [] for k in scores}
    used = 0
    for i, (a, b) in enumerate(pairs):
        faulty = lab.build(8000 + i, [a, b])
        alerts = detect(table_metrics(faulty), stats)
        strengths = {}
        for node, _, z in alerts:
            strengths[node] = max(strengths.get(node, 0.0), min(abs(z), 10) / 10)
        truth = {a["node"], b["node"]}
        detectable = {r for r in truth if r in strengths}
        if not detectable:
            continue
        used += 1
        picks = {
            "ours: peel roots": peel_roots(lab.graph, strengths),
            "baseline: loudest alert": [max(strengths, key=strengths.get)],
            "baseline: most upstream alert": [min(strengths, key=lab.order.index)],
        }
        for method, roots in picks.items():
            recall = len(set(roots) & detectable) / len(detectable)
            precision = len(set(roots) & truth) / len(roots)
            scores[method].append((recall, precision))
            if len(detectable) == 2:
                both[method].append(set(roots) >= detectable)

    results = []
    print(f"{used} double-fault incidents used\n")
    for method, vals in scores.items():
        rec = sum(v[0] for v in vals) / len(vals)
        prec = sum(v[1] for v in vals) / len(vals)
        b = sum(both[method]) / max(len(both[method]), 1)
        results += [(method, "root_cause_recall", rec),
                    (method, "root_cause_precision", prec),
                    (method, "both_causes_found_rate", b)]
        print(f"{method:32} recall {rec:.2f}  precision {prec:.2f}  both found {b:.2f}")
    save_results("multi_fault", results, used)
    print("\nSaved to the metadata store.")


if __name__ == "__main__":
    main()