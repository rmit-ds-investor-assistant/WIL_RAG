import argparse, re
from pathlib import Path
import pandas as pd
from statistics import median, pstdev
from utils import (rouge1, is_unanswered, ndcg_at_k, parse_relevances, tokens, safe_mean)

# Which sheet names map to which cohort (case insensitive)
COHORT_BY_SHEET = {
    "known": "Known",
    "infered": "Inferred",
    "inferred": "Inferred",
    "out_of_kb": "Out of KB",
    "out of kb": "Out of KB",
    "oob": "Out of KB",
}

# Column fallbacks for ground truth and predictions
REF_NAMES = {"gold answer", "reference", "reference answer", "answer"}

def detect_ref_col(df: pd.DataFrame):
    # Prefer exact "Gold Answer", else fallbacks
    for c in df.columns:
        if str(c).strip().lower() in REF_NAMES:
            return c
    # heuristics: any col containing both "gold" and "answer"
    for c in df.columns:
        s = str(c).lower()
        if "gold" in s and "answer" in s:
            return c
    raise ValueError("Missing reference column (e.g., 'Gold Answer').")

def detect_pred_cols(df: pd.DataFrame):
    preds = [c for c in df.columns if re.search(r"generated\s*answer", str(c), re.I)]
    if not preds:
        preds = [c for c in df.columns if re.match(r"(pred|answer_)", str(c), re.I)]
    if not preds:
        raise ValueError("Missing prediction columns (e.g., 'Generated answer k=3').")
    return preds

def load_all_sheets(path: str):
    """Return one concatenated DataFrame with a 'cohort' column derived from sheet names."""
    if not path.lower().endswith((".xlsx", ".xls")):
        # treat as one sheet with no cohort label
        df = pd.read_csv(path).fillna("")
        df["cohort"] = None
        return df

    xls = pd.ExcelFile(path)
    frames = []
    for sname in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sname).fillna("")
        cohort = COHORT_BY_SHEET.get(sname.strip().lower(), None)
        df["cohort"] = cohort
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Excel/CSV with multiple sheets")
    ap.add_argument("--out", default="output_results/metrics_summary.csv")
    ap.add_argument("--round", type=int, default=4)
    ap.add_argument("--auto_oob", action="store_true",
                    help="If no cohort on a row, mark unanswered rows as 'Out of KB' and others as 'Known'.")
    args = ap.parse_args()

    # Load and harmonize
    df = load_all_sheets(args.input)
    df = df.fillna("")

    ref_col = detect_ref_col(df)
    pred_cols = detect_pred_cols(df)

    # Optional: derive cohort if missing
    if args.auto_oob and df["cohort"].isna().any():
        # temporary answered flag for derivation
        tmp_ans = df[pred_cols[0]].apply(lambda x: not is_unanswered(x))
        df.loc[df["cohort"].isna() & ~tmp_ans, "cohort"] = "Out of KB"
        df.loc[df["cohort"].isna() &  tmp_ans, "cohort"] = "Known"

    # Build per-query metrics
    per_rows = []
    for pred_col in pred_cols:
        for _, row in df.iterrows():
            ref, pred = str(row.get(ref_col, "")), str(row.get(pred_col, ""))
            r = rouge1(ref, pred)
            rels = parse_relevances(row)
            nd = ndcg_at_k(rels) if rels else None
            cohort = row.get("cohort", None)

            per_rows.append({
                "variant": pred_col,
                "cohort": cohort,
                "rouge1_p": r["precision"],
                "rouge1_r": r["recall"],
                "rouge1_f1": r["f1"],
                "answered": not is_unanswered(pred),
                "ndcg": nd,
                "pred_len_tok": len(tokens(pred)),
                "ref_len_tok": len(tokens(ref)),
            })

    per_df = pd.DataFrame(per_rows)

    # Variant level summary
    def agg_variant(g):
        f1 = g["rouge1_f1"].tolist()
        return pd.Series({
            "n": len(g),
            "answered_n": int(g["answered"].sum()),
            "answered_%": 100.0 * float(g["answered"].mean()),
            "ROUGE1_P_mean": safe_mean(g["rouge1_p"].tolist()),
            "ROUGE1_R_mean": safe_mean(g["rouge1_r"].tolist()),
            "ROUGE1_F1_mean": safe_mean(f1),
            "ROUGE1_F1_median": float(median(f1)) if f1 else 0.0,
            "ROUGE1_F1_std": float(pstdev(f1)) if len(f1) > 1 else 0.0,
            "NDCG_mean": safe_mean([x for x in g["ndcg"].tolist() if x is not None]),
        })

    sum_df = per_df.groupby("variant", dropna=False).apply(agg_variant).reset_index()
    sum_df = sum_df.sort_values("ROUGE1_F1_mean", ascending=False).reset_index(drop=True)
    sum_ranked = sum_df.copy()
    sum_ranked.insert(0, "rank_by_ROUGE1_F1", range(1, len(sum_ranked)+1))
    num_cols = [c for c in sum_ranked.columns if c not in ["variant","rank_by_ROUGE1_F1","n","answered_n"]]
    sum_ranked[num_cols] = sum_ranked[num_cols].round(args.round)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_df.to_csv(out.parent / "per_query_metrics.csv", index=False)           # detailed diagnostics
    sum_df.to_csv(out.parent / "metrics_summary.csv", index=False)             # raw means
    sum_ranked.to_csv(out.parent / "metrics_summary_ranked.csv", index=False)  # presentation-ready

    # Cohort * Variant cube (for grouped plots)
    if per_df["cohort"].notna().any():
        cube = (per_df
                .groupby(["variant","cohort"], dropna=False)
                .agg(ROUGE1_F1_mean=("rouge1_f1","mean"),
                     NDCG_mean=("ndcg", lambda s: safe_mean([x for x in s if x is not None])),
                     answered_pct=("answered", lambda s: 100.0*float(s.mean())))
                .reset_index())
        cube.to_csv(out.parent / "metrics_by_cohort.csv", index=False)

    print("Saved CSVs to:", out.parent)

if __name__ == "__main__":
    main()