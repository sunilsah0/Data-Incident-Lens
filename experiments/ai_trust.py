from collectors.ai_reliability import (ABSTAIN, ACCESS_LEVEL, DOCS, FABRICATED, MAX_AGE_DAYS,
                                       MIN_SCORE, QUESTIONS, ROLE_LEVEL, TODAY, build_index,
                                       extract_answer, grounding, retrieve)
from experiments.common import save_results

MODES = ["normal", "stale", "leak", "hallucinate"]
ROLES = list(ROLE_LEVEL)
REPLACED = {d["replaces"] for d in DOCS if d["replaces"]}
FAULT_NAME = {"stale": "stale_answers", "leak": "unauthorised_answers",
              "hallucinate": "fabricated_answers"}


def run_rag(question, role, mode):
    doc, score = retrieve(question, role, build_index(mode), mode)
    if doc is not None and score >= MIN_SCORE:
        return dict(text=extract_answer(question, doc), doc=doc, score=score, abstained=False)
    if mode == "hallucinate":
        return dict(text=FABRICATED, doc=None, score=score, abstained=False)
    return dict(text=ABSTAIN, doc=None, score=score, abstained=True)


def grounded(a):
    return a["doc"] is not None and grounding(a["text"], a["doc"]["text"]) >= 0.5


def flag_ours(a, role):
    if a["abstained"]:
        return False
    d = a["doc"]
    if d is None:
        return True                       # an answer that cites no source
    stale = (d["status"] == "superseded" or d["id"] in REPLACED
             or (TODAY - d["updated"]).days > MAX_AGE_DAYS)
    unauthorised = ACCESS_LEVEL[d["access"]] > ROLE_LEVEL[role]
    return stale or unauthorised or not grounded(a)


DETECTORS = {
    "ours: catalogue-aware trust check": flag_ours,
    "baseline: grounding only": lambda a, role: (not a["abstained"]) and not grounded(a),
    "baseline: retrieval score < 0.3": lambda a, role: (not a["abstained"]) and a["score"] < 0.3,
    "baseline: flag every answer": lambda a, role: not a["abstained"],
}


def main():
    questions = sorted({q for q, _ in QUESTIONS})
    rows = []
    for q in questions:
        for role in ROLES:
            gold = run_rag(q, role, "normal")["text"]
            for mode in MODES:
                a = run_rag(q, role, mode)
                rows.append(dict(
                    mode=mode, bad=a["text"] != gold,
                    flags={name: bool(fn(a, role)) for name, fn in DETECTORS.items()}))

    n_bad = sum(r["bad"] for r in rows)
    healthy = [r for r in rows if r["mode"] == "normal"]
    print(f"{len(rows)} answers: {n_bad} bad, {len(healthy)} from the healthy system")
    for m, label in FAULT_NAME.items():
        print(f"  bad {label}: {sum(r['bad'] for r in rows if r['mode'] == m)}")
    print()

    results = []
    for name in DETECTORS:
        tp = sum(r["flags"][name] and r["bad"] for r in rows)
        fp = sum(r["flags"][name] and not r["bad"] for r in rows)
        fn = sum((not r["flags"][name]) and r["bad"] for r in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        false_alarm = sum(r["flags"][name] for r in healthy) / len(healthy)
        results += [(name, "precision", p), (name, "recall", rec), (name, "f1", f1),
                    (name, "false_alarm_rate_on_healthy", false_alarm)]
        line = f"{name:36} precision {p:.2f}  recall {rec:.2f}  f1 {f1:.2f}  false alarms {false_alarm:.2f}"
        for m, label in FAULT_NAME.items():
            bad_rows = [r for r in rows if r["mode"] == m and r["bad"]]
            if bad_rows:
                r_m = sum(r["flags"][name] for r in bad_rows) / len(bad_rows)
                results.append((name, f"recall_{label}", r_m))
                line += f"  | {label} {r_m:.2f}"
        print(line)
    save_results("ai_trust", results, len(rows))
    print("\nSaved to the metadata store.")


if __name__ == "__main__":
    main()