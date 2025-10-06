import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Watercolor theme
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
    })

COHORT_COLORS = {"Known": "#6BAED6", "Inferred": "#74C0B3", "Out of KB": "#B6D7A8"}
BAR_COLOR = "#6BAED6"

# helpers
def prettify_variant(v: str) -> str:
    m = re.search(r"k\s*=\s*(\d+)", str(v), flags=re.I)
    if m:
        return f"k = {int(m.group(1))}"
    s = re.sub(r"^\s*generated\s*answer\s*", "", str(v), flags=re.I).strip()
    return s if s else str(v)

def variant_order_key(v: str):
    m = re.search(r"k\s*=\s*(\d+)", str(v), flags=re.I)
    return (0, int(m.group(1))) if m else (1, str(v).lower())

def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def add_value_labels(ax, bars, y01=False):
    """Place labels smartly to avoid clipping/overlap."""
    for b in bars:
        h = b.get_height()
        if h is None or np.isnan(h):
            continue
        if y01 and h >= 0.95:
            ax.text(b.get_x()+b.get_width()/2, h - 0.03, f"{h:.2f}",
                    ha="center", va="top", fontsize=10, color="#163B65")
        else:
            ax.text(b.get_x()+b.get_width()/2, min(h + 0.02, 1.03) if y01 else h + 1,
                    f"{h:.2f}" if y01 else f"{h:.1f}",
                    ha="center", va="bottom", fontsize=10, color="#1f2d3d")

def finalize(ax, title, ylabel, xlabel="", y01=False, subtitle=None, legend_handles=None):
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if y01:
        ax.set_ylim(0, 1.05)
        ax.margins(y=0.02)
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, ncol=len(legend_handles),
                  loc="upper center", bbox_to_anchor=(0.5, 1.22))
    if subtitle:
        ax.text(0, 1.10, subtitle, transform=ax.transAxes, fontsize=10, color="#4B5B6A")
    plt.tight_layout(rect=(0, 0, 1, 0.88))

# plots
def plot_simple_bars(df, xcol, ycol, title, ylabel, outfile, y01=False):
    set_watercolor_theme()
    df = df.copy()
    df[xcol] = df[xcol].apply(prettify_variant)
    df = df.sort_values(xcol, key=lambda s: s.map(variant_order_key))

    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.bar(df[xcol], df[ycol], color=BAR_COLOR, width=0.55)
    add_value_labels(ax, bars, y01=y01)
    # rotate AFTER setting ticks to avoid warnings
    ax.set_xticks(range(len(df[xcol])), df[xcol], ha="right")
    finalize(ax, title, ylabel, xlabel="Retrieval depth", y01=y01)
    ensure_dir(outfile)
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_grouped(df, metric_col, title, ylabel, outfile, y01=False, subtitle=None):
    set_watercolor_theme()

    # Extract and sort variants
    variants_raw = list(dict.fromkeys(df["variant"]))
    variants_pretty = [prettify_variant(v) for v in variants_raw]
    order = sorted(range(len(variants_pretty)), key=lambda i: variant_order_key(variants_pretty[i]))
    variants_raw = [variants_raw[i] for i in order]
    variants_pretty = [variants_pretty[i] for i in order]

    cohorts = ["Known", "Inferred", "Out of KB"]
    width = 0.22
    xbase = np.arange(len(variants_raw))

    fig, ax = plt.subplots(figsize=(11.5, 6))
    all_bars = []

    for i, c in enumerate(cohorts):
        vals = []
        for v_raw in variants_raw:
            sel = df[(df["variant"] == v_raw) & (df["cohort"] == c)][metric_col]
            vals.append(float(sel.mean()) if not sel.empty else np.nan)
        xs = xbase + (i - 1) * (width + 0.01)
        bars = ax.bar(xs, vals, width=width, label=c, color=COHORT_COLORS[c], edgecolor="white", linewidth=0.8)
        add_value_labels(ax, bars, y01=y01)
        all_bars.append((bars, c))  # store bars + label

    # X labels
    ax.set_xticks(xbase)
    ax.set_xticklabels(variants_pretty, rotation=15, ha="right")

    # Legend fix: use manual handles and real labels
    handles = [plt.Rectangle((0, 0), 1, 1, color=COHORT_COLORS[c], label=c) for c in cohorts]
    ax.legend(handles=handles, frameon=False, ncol=len(cohorts),
              loc="upper center", bbox_to_anchor=(0.5, 1.20))

    # Title and axis adjustments
    ylim = (0, 1.05) if y01 else None
    finalize(ax, title, ylabel, xlabel="Retrieval depth", y01=y01, subtitle=subtitle)
    ensure_dir(outfile)
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

# CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to output_results/metrics_summary_ranked.csv OR metrics_by_cohort.csv")
    ap.add_argument("--out", default="output_results/plots.png")
    args = ap.parse_args()
    out_stub = args.out.replace(".png", "")

    if "metrics_by_cohort" in args.input:
        c = pd.read_csv(args.input)

        if "ROUGE1_F1_mean" in c.columns:
            plot_grouped(
                c, "ROUGE1_F1_mean",
                title="ROUGE-1 (mean) by Cohort",
                ylabel="ROUGE-1 (mean)",
                outfile=f"{out_stub}_rouge1_by_cohort.png",
                y01=True,
                subtitle="Higher is better"
            )

        if "NDCG_mean" in c.columns and not pd.Series(c["NDCG_mean"]).isna().all():
            plot_grouped(
                c, "NDCG_mean",
                title="NDCG (mean) for Known and Inferred",
                ylabel="NDCG (mean)",
                outfile=f"{out_stub}_ndcg_by_cohort.png",
                y01=True,
                subtitle="Higher is better"
            )

        if "answered_pct" in c.columns:
            c_pct = c.copy()
            c_pct["answered_frac"] = c_pct["answered_pct"] / 100.0
            plot_grouped(
                c_pct, "answered_frac",
                title="Answer Coverage (%) by Cohort",
                ylabel="% Answered",
                outfile=f"{out_stub}_answered_pct_by_cohort.png",
                y01=True,
                subtitle="Closer to 1 (100%) is better"
            )

    else:
        s = pd.read_csv(args.input).sort_values("ROUGE1_F1_mean", ascending=False)

        plot_simple_bars(
            s, "variant", "ROUGE1_F1_mean",
            title="ROUGE-1 F1 by Retrieval Depth",
            ylabel="ROUGE-1 F1",
            outfile=f"{out_stub}_rouge1_f1_by_depth.png",
            y01=True
        )

        s_frac = s.copy(); s_frac["answered_frac"] = s_frac["answered_%"] / 100.0
        plot_simple_bars(
            s_frac, "variant", "answered_frac",
            title="Answer Coverage (%)",
            ylabel="% Answered",
            outfile=f"{out_stub}_answered_pct.png",
            y01=True
        )

        if "NDCG_mean" in s.columns and not pd.Series(s["NDCG_mean"]).isna().all():
            plot_simple_bars(
                s, "variant", "NDCG_mean",
                title="NDCG by Retrieval Depth",
                ylabel="NDCG (mean)",
                outfile=f"{out_stub}_ndcg_by_depth.png",
                y01=True
            )

    print("Saved plots with prefix:", f"{out_stub}_*.png")

if __name__ == "__main__":
    main()