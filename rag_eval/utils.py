import re, json, math, string
from collections import Counter
from statistics import mean

# Unanswered detection
UNK_TOKENS = {
    "", "n/a", "na", "none", "no answer", "null", "unknown",
    "i don't know", "idk", "not sure", "cannot answer", "can't answer",
    "insufficient information", "not enough information"
}
MIN_CHAR = 3
UNK_REGEXES = [
    r"\b(i|we)\s+(don'?t|do not)\s+know\b",
    r"\b(cannot|can't)\s+answer\b",
    r"\binsufficient\s+information\b",
    r"\bnot\s+enough\s+information\b"
]

# Text helpers
def normalize_text(s):
    s = "" if s is None else str(s)
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()

def tokens(s):
    t = normalize_text(s).split(" ")
    return [x for x in t if x]

def rouge1(ref, pred):
    ref_t, pred_t = tokens(ref), tokens(pred)
    if not ref_t and not pred_t:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    rc, pc = Counter(ref_t), Counter(pred_t)
    overlap = sum(min(rc[w], pc[w]) for w in set(rc) | set(pc))
    p = overlap / max(len(pred_t), 1)
    r = overlap / max(len(ref_t), 1)
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f1}

def is_unanswered(pred):
    s = "" if pred is None else str(pred).strip().lower()
    if len(s) < MIN_CHAR or s in UNK_TOKENS:
        return True
    return any(re.search(rx, s) for rx in UNK_REGEXES)

# NDCG
def _dcg(rels):
    return sum(((2**rel - 1) / math.log2(i + 2)) for i, rel in enumerate(rels))

def ndcg_at_k(rels):
    if not rels: return None
    ideal = sorted(rels, reverse=True)
    den = _dcg(ideal)
    return 0.0 if den == 0 else _dcg(rels) / den

def parse_relevances(row):
    # JSON list in 'retrieved_relevance'
    if "retrieved_relevance" in row and row["retrieved_relevance"]:
        try:
            vals = row["retrieved_relevance"]
            if isinstance(vals, str): vals = json.loads(vals)
            return [int(v) for v in vals if str(v).strip().lstrip("-").isdigit()]
        except Exception:
            pass
    # rel_1..rel_n columns
    cols = [c for c in row.index if str(c).lower().startswith("rel_")]
    rels = []
    for c in cols:
        v = row.get(c, "")
        try:
            rels.append(int(v))
        except Exception:
            pass
    return rels if rels else None

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return 0.0 if not xs else float(mean(xs))