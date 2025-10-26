import os
import re
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

_ABSTAIN_PATTERNS = [
    r"\b(i (do not|don't) know)\b",
    r"\b(not (in|within) (the )?(kb|knowledge base|documents|docs))\b",
    r"\b(cannot|can't) (find|locate)\b",
    r"\b(no (information|data) (available|found))\b",
    r"\b(insufficient (context|information))\b",
    r"\bout[- ]of[- ]kb\b",
    r"\b(i cannot (answer|provide))\b",
    r"\b(no (knowledge|record) of this)\b",
]

def clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip())

def looks_like_refusal(s: str) -> bool:
    s_low = clean(s).lower()
    return any(re.search(p, s_low) for p in _ABSTAIN_PATTERNS)

_rouge = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
_sbert: Optional[SentenceTransformer] = None

def _lazy_sbert():
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sbert

def rougeL_f1(pred: str, ref: str) -> float:
    pred, ref = clean(pred), clean(ref)
    if not pred or not ref:
        return 0.0
    return _rouge.score(ref, pred)["rougeLsum"].fmeasure

def sbert_cosine(pred: str, ref: str) -> float:
    pred, ref = clean(pred), clean(ref)
    if not pred or not ref:
        return 0.0
    model = _lazy_sbert()
    emb = model.encode([pred, ref], convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(emb[0], emb[1]).item())

def score_known_infer(gold: str, pred: str) -> Dict[str, float]:
    return {
        "rougeL": rougeL_f1(pred, gold),
        "semantic_cosine": sbert_cosine(pred, gold),
        "good_refusal": np.nan,
        "hallucination_rate": np.nan,
    }

def score_outkb(pred: str) -> Dict[str, float]:
    refusal = looks_like_refusal(pred)
    answered_nonempty = bool(clean(pred))
    hallucination = int(answered_nonempty and not refusal)
    return {
        "rougeL": np.nan,
        "semantic_cosine": np.nan,
        "good_refusal": int(refusal),
        "hallucination_rate": hallucination,
    }

def set_watercolor_theme():
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "#F7FAFC",
        "axes.facecolor":   "#F7FAFC",
        "savefig.facecolor":"#F7FAFC",
        "axes.edgecolor":   "#9FB3C8",
        "axes.labelcolor":  "#213547",
        "text.color":       "#213547",
        "xtick.color":      "#213547",
        "ytick.color":      "#213547",
        "grid.color":       "#D7E3EE",
        "grid.linestyle":   "-",
        "grid.linewidth":   0.8,
        "axes.grid":        True,
        "font.size":        12,
        "axes.titleweight": "semibold",
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Lato", "DejaVu Sans", "Helvetica", "Arial"],
    })