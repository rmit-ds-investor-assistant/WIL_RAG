import argparse
import os
import re
import pandas as pd
from tqdm import tqdm

from utils import ensure_dirs, clean, score_known_infer, score_outkb

COHORT_KNOWN   = "Known"
COHORT_INFER   = "Inferred"     # normalize spelling
COHORT_OUTKB   = "Out of KB"

COLS_COMMON_K = {
    1: "Generated answer k=1",
    3: "Generated answer k=3",
    5: "Generated answer k=5",
}
COLS_BASE = {
    "query": "Query",
    "expected_company": "Expected Company",
    "gold": "Gold Answer",
}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _build_sheet_map(xls: pd.ExcelFile) -> dict:
    return {_norm(name): name for name in xls.sheet_names}

def _find_sheet(xls_map: dict, aliases: list[str]) -> str | None:
    for a in aliases:
        k = _norm(a)
        if k in xls_map:
            return xls_map[k]
    return None

ALIASES_KNOWN  = ["Known", " known ", "KNOWN"]
ALIASES_INFER  = ["Infered", "Inferred"]
ALIASES_OUTKB  = [
    "Out of KB", "Out-of-KB", "Out of Knowledge Base",
    "OutOfKB", "OutKB", "Out of Kb", "Out of K B"
]

def process_known_or_infered(df: pd.DataFrame, cohort_label: str) -> pd.DataFrame:
    rows = []
    for i, r in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {cohort_label}"):
        gold  = clean(r.get(COLS_BASE["gold"], ""))
        query = clean(r.get(COLS_BASE["query"], ""))
        expco = clean(r.get(COLS_BASE["expected_company"], ""))
        for k in (1, 3, 5):
            pred = r.get(COLS_COMMON_K[k], "")
            met = score_known_infer(gold, pred)
            met.update({
                "cohort": cohort_label,
                "row_id": i,
                "k": k,
                "query": query,
                "expected_company": expco,
                "gold": gold,
                "pred": clean(pred),
            })
            rows.append(met)
    return pd.DataFrame(rows)

def process_outkb(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, r in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {COHORT_OUTKB}"):
        for k in (1, 3, 5):
            pred = r.get(COLS_COMMON_K[k], "")
            met = score_outkb(pred)
            met.update({
                "cohort": COHORT_OUTKB,
                "row_id": i,
                "k": k,
                "pred": clean(pred),
                "query": "",
                "expected_company": "",
                "gold": "",
            })
            rows.append(met)
    return pd.DataFrame(rows)

def aggregate(per_row: pd.DataFrame) -> pd.DataFrame:
    agg_cols = {
        "rougeL": "mean",
        "semantic_cosine": "mean",
        "good_refusal": "mean",
        "hallucination_rate": "mean",
    }
    return (
        per_row
        .groupby(["cohort", "k"], dropna=False)
        .agg(agg_cols)
        .reset_index()
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="data/RAGfaq.xlsx")
    ap.add_argument("--outdir", "-o", default="output_results")
    args = ap.parse_args()

    ensure_dirs(args.outdir)

    xls = pd.ExcelFile(args.input)
    smap = _build_sheet_map(xls)

    sheet_known  = _find_sheet(smap, ALIASES_KNOWN)
    sheet_infer  = _find_sheet(smap, ALIASES_INFER)
    sheet_outkb  = _find_sheet(smap, ALIASES_OUTKB)

    frames = []

    if sheet_known:
        frames.append(process_known_or_infered(pd.read_excel(args.input, sheet_name=sheet_known), COHORT_KNOWN))
    if sheet_infer:
        frames.append(process_known_or_infered(pd.read_excel(args.input, sheet_name=sheet_infer), COHORT_INFER))
    if sheet_outkb:
        frames.append(process_outkb(pd.read_excel(args.input, sheet_name=sheet_outkb)))

    if not frames:
        raise RuntimeError(
            "No recognized sheets found. Available: "
            + ", ".join(xls.sheet_names)
        )

    per_row = pd.concat(frames, ignore_index=True)
    per_row.to_csv(os.path.join(args.outdir, "metrics_per_query.csv"), index=False)

    by_cohort = aggregate(per_row)
    by_cohort.to_csv(os.path.join(args.outdir, "metrics_by_cohort.csv"), index=False)

    print("[OK] Wrote:")
    print(" -", os.path.join(args.outdir, "metrics_per_query.csv"))
    print(" -", os.path.join(args.outdir, "metrics_by_cohort.csv"))

if __name__ == "__main__":
    main()