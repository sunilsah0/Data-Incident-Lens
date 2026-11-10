import random
import re
import sys
from datetime import date, datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from engine.store import get_store, init_store, register_asset

ASSET = "ai.support_rag"
TODAY = date(2024, 2, 1)      # fixed "today", so results are reproducible
MAX_AGE_DAYS = 365
MIN_SCORE = 0.15              # below this, the best document is not relevant
N_QUESTIONS = 20
ROLE_LEVEL = {"customer": 0, "driver": 1, "manager": 2}
ACCESS_LEVEL = {"public": 0, "internal": 1, "restricted": 2}

DOCS = [
    dict(id="pricing_v1", updated=date(2022, 6, 1), status="superseded", replaces=None, access="public",
         text="The base fare is 2 dollars. The price per mile is 2 dollars. The night surcharge is 1 dollar between 10pm and 6am."),
    dict(id="pricing_v2", updated=date(2024, 1, 1), status="current", replaces="pricing_v1", access="public",
         text="The base fare is 3 dollars. The price per mile is 4 dollars. The night surcharge is 2 dollars between 10pm and 6am."),
    dict(id="airport_v1", updated=date(2022, 6, 1), status="superseded", replaces=None, access="public",
         text="The airport pickup fee is 4 dollars. Airport pickups use the main terminal lane."),
    dict(id="airport_v2", updated=date(2024, 1, 5), status="current", replaces="airport_v1", access="public",
         text="The airport pickup fee is 8 dollars. Airport pickups use the new terminal lane."),
    dict(id="refund_policy", updated=date(2024, 1, 10), status="current", replaces=None, access="public",
         text="Customers get a full refund if the driver cancels a trip. Refund requests must be sent within 14 days. Refunds are paid to the original payment method within 5 working days."),
    dict(id="driver_payout", updated=date(2024, 1, 8), status="current", replaces=None, access="internal",
         text="Drivers are paid every Friday. The platform keeps 20 percent of each fare. Drivers keep 100 percent of tips."),
    dict(id="data_retention", updated=date(2024, 1, 12), status="current", replaces=None, access="internal",
         text="Customer personal data such as phone numbers and emails is kept for 90 days after the last trip. After that the data is deleted."),
    dict(id="exec_salaries", updated=date(2024, 1, 15), status="current", replaces=None, access="restricted",
         text="Executive salary bands: the chief executive earns between 300000 and 400000 dollars. Directors earn between 150000 and 220000 dollars."),
]

QUESTIONS = [
    ("What is the base fare and price per mile?", "customer"),
    ("What is the night surcharge?", "customer"),
    ("How much is the airport pickup fee?", "customer"),
    ("How do I get a refund when the driver cancels?", "customer"),
    ("How many days do I have to request a refund?", "customer"),
    ("When are drivers paid and how much does the platform keep?", "driver"),
    ("How long is customer personal data kept?", "customer"),
    ("What are the executive salary bands?", "driver"),
    ("What is the salary range for directors?", "customer"),
    ("Can I bring my pet dog in the taxi?", "customer"),
    ("Do you offer helicopter rides?", "customer"),
    ("Is there a discount for students?", "customer"),
]

ABSTAIN = "I do not have information about that."
FABRICATED = "Yes, this is available at most locations and handled according to our standard procedure."


def words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3}


def build_index(mode):
    if mode == "stale":
        # the index was never refreshed: old versions stay, new versions are missing
        return [d for d in DOCS if d["replaces"] is None]
    return [d for d in DOCS if d["status"] == "current"]


def retrieve(question, role, index, mode):
    candidates = index
    if mode != "leak":   # the access filter
        candidates = [d for d in index if ACCESS_LEVEL[d["access"]] <= ROLE_LEVEL[role]]
    if not candidates:
        return None, 0.0
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform([d["text"] for d in candidates])
    scores = cosine_similarity(vec.transform([question]), matrix)[0]
    best = int(scores.argmax())
    return candidates[best], float(scores[best])


def extract_answer(question, doc):
    sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
    q = words(question)
    ranked = sorted(sentences, key=lambda s: len(words(s) & q), reverse=True)
    return " ".join(ranked[:2])


def grounding(answer, doc_text):
    a = words(answer)
    if not a or doc_text is None:
        return 0.0
    return len(a & words(doc_text)) / len(a)


def collect_ai(mode="normal"):
    init_store()
    register_asset(ASSET, "rag_index", "support_rag", 3)
    con = get_store()
    con.execute("""
        CREATE TABLE IF NOT EXISTS rag_answers (
            ts TIMESTAMP, mode VARCHAR, question VARCHAR, role VARCHAR,
            doc_id VARCHAR, score DOUBLE, answer VARCHAR, abstained BOOLEAN,
            stale BOOLEAN, unauthorized BOOLEAN, ungrounded BOOLEAN)""")

    index = build_index(mode)
    now = datetime.now()
    rows = []
    for question, role in random.choices(QUESTIONS, k=N_QUESTIONS):
        doc, score = retrieve(question, role, index, mode)
        if doc is not None and score >= MIN_SCORE:
            answer, used, abstained = extract_answer(question, doc), doc, False
        elif mode == "hallucinate":
            answer, used, abstained = FABRICATED, None, False
        else:
            answer, used, abstained = ABSTAIN, None, True

        stale = used is not None and (
            used["status"] == "superseded" or (TODAY - used["updated"]).days > MAX_AGE_DAYS)
        unauthorized = used is not None and ACCESS_LEVEL[used["access"]] > ROLE_LEVEL[role]
        ungrounded = (not abstained) and grounding(answer, used["text"] if used else None) < 0.5
        rows.append((now, mode, question, role, used["id"] if used else None, score,
                     answer, abstained, stale, unauthorized, ungrounded))
    con.executemany("INSERT INTO rag_answers VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)

    answered = [r for r in rows if not r[7]]
    n = max(len(answered), 1)
    metrics = {
        "stale_answer_rate": sum(r[8] for r in answered) / n,
        "unauthorized_answer_rate": sum(r[9] for r in answered) / n,
        "ungrounded_answer_rate": sum(r[10] for r in answered) / n,
        "abstain_rate": sum(r[7] for r in rows) / len(rows),
        "avg_retrieval_score": sum(r[5] for r in answered) / n,
    }
    print(f"Mode: {mode}")
    for name, value in metrics.items():
        bad = name.endswith("_answer_rate") and value > 0.05
        status = "fail" if bad else "ok"
        con.execute(
            "INSERT INTO events (ts, asset_id, module, metric, value, status, details) "
            "VALUES (?, ?, 'ai', ?, ?, ?, ?)", [now, ASSET, name, value, status, mode])
        print(f"  {name:26} {value:7.3f}  [{status}]")
    con.close()


if __name__ == "__main__":
    mode = "normal"
    for flag in ("stale", "leak", "hallucinate"):
        if f"--{flag}" in sys.argv:
            mode = flag
    collect_ai(mode)