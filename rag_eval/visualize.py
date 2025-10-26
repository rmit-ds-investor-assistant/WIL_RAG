import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import ensure_dirs, set_watercolor_theme

PRETTY = {
    "rougeL": "ROUGE-L",
    "semantic_cosine": "Semantic similarity (SBERT cosine)",
}

QUALITY = ["rougeL", "semantic_cosine"]  # Known + Inferred only

# --------------------------
# Helpers
# --------------------------
def ci95(std: pd.Series, n: pd.Series) -> np.ndarray:
    return 1.96 * (std / np.sqrt(n).replace(0, np.nan))

def shared_ylim(per_row: pd.DataFrame, metric: str, pad=0.06):
    s = per_row[metric].dropna()
    if s.empty:
        return None
    lo, hi = np.nanquantile(s, [0.02, 0.98])
    rng = hi - lo if hi > lo else 1.0
    return max(0.0, lo - pad*rng), min(1.0, hi + pad*rng)

def save_png(fig, out_path_png: str, dpi: int = 300):
    fig.savefig(out_path_png, dpi=dpi)
    plt.close(fig)

def jitter_points(ax, sub: pd.DataFrame, metric: str, jitter=0.08, s=26, alpha=0.25):
    for k, grp in sub.groupby("k"):
        vals = grp[metric].dropna().values
        if len(vals) == 0:
            continue
        xs = np.full(len(vals), float(k)) + np.random.uniform(-jitter, jitter, len(vals))
        ax.scatter(xs, vals, s=s, alpha=alpha, edgecolors="none")

# --------------------------
# Trends: small multiples (with jitter + CI ribbons + in-panel annotations)
# --------------------------
def trend_small_multiples(per_row: pd.DataFrame, metric: str, outdir: str,
                          panel_w=6.4, panel_h=4.9):
    set_watercolor_theme()

    cohorts = sorted(per_row.loc[per_row[metric].notna(), "cohort"].unique().tolist())
    if not cohorts:
        return None

    # order panels by baseline mean (k=1)
    ordered = []
    for c in cohorts:
        base = per_row[(per_row["cohort"]==c) & (per_row["k"]==1)][metric].mean()
        ordered.append((c, np.inf if pd.isna(base) else base))
    cohorts = [c for c,_ in sorted(ordered, key=lambda x: x[1])]

    cols = min(3, len(cohorts))
    rows = int(np.ceil(len(cohorts) / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(panel_w*cols, panel_h*rows),
                             dpi=300, squeeze=False)

    ylim = shared_ylim(per_row[per_row["cohort"].isin(cohorts)], metric, pad=0.08)
    fig.suptitle(PRETTY.get(metric, metric), y=0.98, fontsize=16)

    for i, cohort in enumerate(cohorts):
        ax = axes[i//cols, i%cols]
        sub = per_row[(per_row["cohort"] == cohort) & (per_row[metric].notna())][["k", metric]]

        g = sub.groupby("k")[metric].agg(["mean", "std", "count"]).sort_index()
        if g.empty or g["count"].sum() == 0:
            ax.axis("off"); continue

        xs = g.index.values.astype(float)
        ys = g["mean"].values
        yerr = ci95(g["std"], g["count"]).values

        # raw points (jittered)
        jitter_points(ax, sub, metric)

        # mean + 95% CI ribbon
        ax.fill_between(xs, ys - yerr, ys + yerr, alpha=0.18)
        ax.plot(xs, ys, marker="o", linewidth=2)

        ax.set_xlabel("k (top-k retrieved)")
        ax.set_ylabel(PRETTY.get(metric, metric))
        ax.set_title(cohort, loc="left", fontsize=13)
        if ylim: ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.35)

        # Δ(1→5), last value, and n-per-k annotations
        if 1 in g.index and 5 in g.index and np.isfinite(g.loc[1,"mean"]) and np.isfinite(g.loc[5,"mean"]):
            delta = g.loc[5, "mean"] - g.loc[1, "mean"]
            ax.text(0.02, 0.02, f"Δ(1→5) = {delta:.02f}", transform=ax.transAxes, fontsize=10)
        ax.text(xs[-1] + 0.06, ys[-1], f"{ys[-1]:.2f}", va="center", fontsize=10)

        span = (ylim[1]-ylim[0]) if ylim else 0.2
        voff = span * 0.04
        for xv, yv, cnt in zip(xs, ys, g["count"].values):
            ax.text(xv, yv - voff, f"n={int(cnt)}", ha="center", va="top", fontsize=9, alpha=0.85)

    # turn off unused panels
    for j in range(i+1, rows*cols):
        axes[j//cols, j%cols].axis("off")

    fig.tight_layout(rect=[0,0,1,0.96])
    out_path = os.path.join(outdir, f"trend_small_multiples_{metric}.png")
    save_png(fig, out_path)
    return out_path

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perrow", "-r", default="output_results/metrics_per_query.csv")
    ap.add_argument("--outdir", "-o", default="plots")
    ap.add_argument("--panelw", type=float, default=6.4)
    ap.add_argument("--panelh", type=float, default=4.9)
    args = ap.parse_args()

    ensure_dirs(args.outdir)
    per_row = pd.read_csv(args.perrow)

    # Clean cohort labels for display
    per_row["cohort"] = per_row["cohort"].replace({"Infered": "Inferred"})

    made = []
    for m in QUALITY:
        if m in per_row.columns and per_row[m].notna().any():
            made.append(trend_small_multiples(per_row, m, args.outdir,
                                              panel_w=args.panelw, panel_h=args.panelh))

    if made:
        print("[OK] Plots:")
        for p in made:
            print("  -", p)
    else:
        print("[WARN] No informative plots generated.")

if __name__ == "__main__":
    main()