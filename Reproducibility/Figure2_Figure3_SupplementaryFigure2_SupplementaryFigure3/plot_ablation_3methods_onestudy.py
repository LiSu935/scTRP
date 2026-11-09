#!/usr/bin/env python3
"""
plot_ablation_3methods_onestudy.py

Simple 3-method comparison (scTRP vs scGPT vs XGBoost) on a single study.
Cell-level and clonotype-level patient-wise metrics (MCC, Gmean, ROC-AUC),
each as one grouped barplot: x-axis = the metrics, 3 bars/metric = methods,
per-patient values overlaid as jittered scatter, paired t-test significance
brackets for (scTRP vs scGPT) and (scTRP vs XGBoost).

Inputs
------
--sctrp_csv    columns: cell_barcode, score, reactivity, sample_clone_id, sample, reactivity_fine
--xgboost_csv  columns: cell_barcode, reactivity, xgboost_score, xgboost_pred
--scgpt_h5ad   .obs columns: reactivity, predictions, score  (index = cell_barcode)

scTRP is the sample / sample_clone_id source of truth: XGBoost and scGPT cells
are matched to it by cell_barcode. Cell-level prediction is the method's own
'pred' column if present (XGBoost, scGPT), else per-patient Jenks(k=2) on
'score' (scTRP). Clonotype-level always max-pools 'score' per sample_clone_id
to one row per clonotype, then re-derives 'pred' via per-patient Jenks on the
pooled score for every method (even XGBoost/scGPT).

Usage
-----
python plot_ablation_3methods_onestudy.py \\
    --sctrp_csv /path/scTRP_predictions.csv \\
    --xgboost_csv /path/xgboost_predictions.csv \\
    --scgpt_h5ad /path/scgpt_predictions.h5ad \\
    --out_dir /path/out
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ttest_rel as _ttest_rel_fn
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

try:
    import jenkspy
    HAS_JENKSPY = True
except ImportError:
    HAS_JENKSPY = False
    print("[warn] jenkspy not found — using median fallback for Jenks breaks")

warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          16,
    "axes.labelsize":     20,
    "axes.titlesize":     22,
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "xtick.labelsize":    20,
    "ytick.labelsize":    17,
    "legend.fontsize":    17,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.linewidth":     1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "xtick.major.size":   5,
    "ytick.major.size":   5,
    "lines.linewidth":    2.0,
})

# ── Constants ─────────────────────────────────────────────────────────────────
METHOD_ORDER = ["scTRP", "scGPT", "xgboost"]
METHOD_LABELS = {"scTRP": "scTRP", "scGPT": "scGPT", "xgboost": "XGBoost"}
METHOD_PALETTE = {"scTRP": "#d62728", "scGPT": "#1f77b4", "xgboost": "#888888"}
METRICS = [
    ("ROC_AUC", "ROC-AUC"),
    ("MCC",     "MCC"),
    ("Gmean",   "G-mean"),
]
REF_METHOD = "scTRP"
COMPARE_TO = ["scGPT", "xgboost"]


# ── Metric helpers ────────────────────────────────────────────────────────────

def _jenks_thresh(scores):
    if HAS_JENKSPY and len(scores) >= 2:
        try:
            return jenkspy.jenks_breaks(list(scores), n_classes=2)[1]
        except Exception:
            pass
    return float(np.median(scores))


def compute_patient_metrics(labels, preds, scores):
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())

    result = {"MCC": np.nan, "Gmean": np.nan, "ROC_AUC": np.nan,
              "n_cells": len(labels), "n_pos": n_pos, "n_neg": n_neg}
    if n_pos == 0 or n_neg == 0:
        return result
    try:
        result["MCC"] = matthews_corrcoef(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        result["Gmean"] = (sen * spe) ** 0.5
        result["ROC_AUC"] = roc_auc_score(labels, scores)
    except Exception:
        pass
    return result


def compute_patient_wise_from_df(df):
    """Per-patient MCC/Gmean/ROC-AUC. Uses df['pred'] if present (and fully
    populated), else per-patient Jenks(k=2) on df['score']. df needs
    score, reactivity, sample columns."""
    has_pred = "pred" in df.columns and df["pred"].notna().any()
    rows = []
    for samp, grp in df.groupby("sample"):
        g = grp.dropna(subset=["score", "reactivity"])
        if len(g) < 2:
            continue
        scores = g["score"].values.astype(float)
        labels = g["reactivity"].values.astype(int)
        if has_pred and g["pred"].notna().all():
            preds = g["pred"].values.astype(int)
        else:
            try:
                thresh = _jenks_thresh(scores)
                preds = (scores > thresh).astype(int)
            except Exception as e:
                print(f"    skip {samp}: jenks failed ({e})")
                continue
        m = compute_patient_metrics(labels, preds, scores)
        m["sample"] = samp
        rows.append(m)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _ttest_stars(a, b, alternative="greater"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == b):
        return np.nan, "ns"
    try:
        _, pval = _ttest_rel_fn(a, b, alternative=alternative)
        star = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        return float(pval), star
    except Exception:
        return np.nan, ""


def _sig_bracket(ax, x1, x2, y, h=0.025, star="", fontsize=22,
                  color="#333333", lw=1.2, show_ns=False):
    if not star or (star == "ns" and not show_ns):
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, color=color, clip_on=False)
    fw = "bold" if star != "ns" else "normal"
    ax.text((x1 + x2) / 2, y + h + 0.003, star, ha="center", va="bottom",
             fontsize=fontsize, color=color, fontweight=fw, clip_on=False)


# ── Data loading / harmonization ───────────────────────────────────────────────

def load_sctrp(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "cell_barcode"
    df["reactivity"] = pd.to_numeric(df["reactivity"], errors="coerce")
    return df[["score", "reactivity", "sample_clone_id", "sample"]].copy()


def load_xgboost(csv_path, barcode_map):
    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "cell_barcode"
    df["reactivity"] = pd.to_numeric(df["reactivity"], errors="coerce")
    df = df.rename(columns={"xgboost_score": "score", "xgboost_pred": "pred"})
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["sample"] = barcode_map["sample"].reindex(df.index)
    df["sample_clone_id"] = barcode_map["sample_clone_id"].reindex(df.index)
    return df[["score", "reactivity", "pred", "sample_clone_id", "sample"]]


def load_scgpt(h5ad_path, barcode_map):
    adata = sc.read_h5ad(h5ad_path)
    obs = adata.obs.copy()
    obs.index.name = "cell_barcode"
    df = pd.DataFrame(index=obs.index)
    df["score"] = pd.to_numeric(obs["score"], errors="coerce")
    df["reactivity"] = pd.to_numeric(obs["reactivity"], errors="coerce")
    df["pred"] = pd.to_numeric(obs["predictions"], errors="coerce")
    df["sample"] = barcode_map["sample"].reindex(df.index)
    df["sample_clone_id"] = barcode_map["sample_clone_id"].reindex(df.index)
    return df


def report_and_drop_unmatched(df, method_name):
    n_before = len(df)
    df = df.dropna(subset=["sample"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [{method_name}] dropped {n_dropped}/{n_before} cells with no scTRP "
              f"barcode match (no sample/sample_clone_id)")
    return df


def pool_clonotype_max(df):
    """One row per sample_clone_id: max-pooled score, reactivity/sample taken
    as first (assumed constant within a clonotype). No 'pred' column carried
    over — clonotype-level prediction is always re-derived via Jenks."""
    d = df.dropna(subset=["sample_clone_id", "score", "reactivity", "sample"])
    pooled = d.groupby("sample_clone_id").agg(
        score=("score", "max"), reactivity=("reactivity", "first"), sample=("sample", "first")
    )
    return pooled.reset_index(drop=True)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_grouped_metric_barplot(df_patient, out_path, title, alternative="greater"):
    """Returns a list of dicts with the paired t-test stats for every
    (metric, ref_method, other_method) bracket drawn — save these to CSV
    alongside the figure."""
    methods = [m for m in METHOD_ORDER if m in df_patient["method"].unique()]
    metric_cols = [c for c, _ in METRICS if c in df_patient.columns]
    metric_labels = [l for c, l in METRICS if c in metric_cols]
    if not methods or not metric_cols:
        print(f"  [skip] {title}: no methods/metrics to plot")
        return []

    n_meth = len(methods)
    bar_w = 0.80 / n_meth
    rng = np.random.RandomState(42)
    x = np.arange(len(metric_cols))

    fig, ax = plt.subplots(figsize=(2.4 * len(metric_cols) + 2.5, 6.5), constrained_layout=True)

    pat_by_method = {m: df_patient[df_patient["method"] == m].set_index("sample") for m in methods}

    for i, method in enumerate(methods):
        offset = (i - n_meth / 2 + 0.5) * bar_w
        vals = [pat_by_method[method][c].mean() if c in pat_by_method[method] else np.nan
                for c in metric_cols]
        ax.bar(x + offset, vals, width=bar_w * 0.88,
               color=METHOD_PALETTE.get(method, "#aaa"), alpha=0.72,
               edgecolor="white", linewidth=0.5,
               label=METHOD_LABELS.get(method, method), zorder=3)

    y_group_max = {c: -np.inf for c in metric_cols}
    y_group_min = {c: np.inf for c in metric_cols}
    for i, method in enumerate(methods):
        offset = (i - n_meth / 2 + 0.5) * bar_w
        for j, c in enumerate(metric_cols):
            yvals = pat_by_method[method][c].dropna().values if c in pat_by_method[method] else np.array([])
            if len(yvals) == 0:
                continue
            jitter = rng.uniform(-bar_w * 0.27, bar_w * 0.27, size=len(yvals))
            ax.scatter(j + offset + jitter, yvals, color=METHOD_PALETTE.get(method, "#aaa"),
                       s=42, alpha=0.88, edgecolors="black", linewidths=0.5, zorder=6)
            y_group_max[c] = max(y_group_max[c], float(np.nanmax(yvals)))
            y_group_min[c] = min(y_group_min[c], float(np.nanmin(yvals)))

    finite_max = [v for v in y_group_max.values() if np.isfinite(v)]
    finite_min = [v for v in y_group_min.values() if np.isfinite(v)]
    all_max = max(finite_max) if finite_max else 1.0
    all_min = min(finite_min) if finite_min else 0.0
    compare_to = [m for m in COMPARE_TO if m in methods]
    top = all_max + 0.08 + max(len(compare_to), 1) * 0.10
    bottom = min(0.0, all_min) - 0.05
    ax.set_ylim(bottom, top)

    stats_rows = []
    if REF_METHOD in methods:
        idx_ref = methods.index(REF_METHOD)
        pa_full = pat_by_method[REF_METHOD]
        for k, other in enumerate(compare_to):
            idx_other = methods.index(other)
            pb_full = pat_by_method[other]
            for j, c in enumerate(metric_cols):
                common = pa_full.index.intersection(pb_full.index)
                a = pa_full.loc[common, c].dropna()
                b = pb_full.loc[common, c].dropna()
                common2 = a.index.intersection(b.index)
                n_patients = len(common2)
                if n_patients < 3:
                    continue
                a_vals = a.loc[common2].values.astype(float)
                b_vals = b.loc[common2].values.astype(float)
                pval, star = _ttest_stars(a_vals, b_vals, alternative=alternative)
                stats_rows.append({
                    "metric": c, "ref_method": REF_METHOD, "other_method": other,
                    "alternative": alternative, "n_patients": n_patients,
                    "ref_mean": float(a_vals.mean()), "other_mean": float(b_vals.mean()),
                    "mean_diff": float(a_vals.mean() - b_vals.mean()),
                    "pval": pval, "star": star,
                })
                x_a = j + (idx_ref - n_meth / 2 + 0.5) * bar_w
                x_b = j + (idx_other - n_meth / 2 + 0.5) * bar_w
                y_base = y_group_max[c] + 0.05 + k * 0.09
                _sig_bracket(ax, min(x_a, x_b), max(x_a, x_b), y_base,
                             h=0.025, star=star, fontsize=22)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=20)
    ax.tick_params(axis="y", labelsize=17)
    ax.axhline(0, color="lightgray", lw=0.8, ls="--", zorder=1)
    ax.set_ylabel("Score", fontsize=20)
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_PALETTE.get(m, "#aaa"), alpha=0.72)
               for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False, fontsize=17)
    if title:
        ax.set_title(title, fontsize=22, pad=55)

    fig.show()
    fig.savefig(f"{out_path}.pdf")
    fig.savefig(f"{out_path}.png", dpi=300)
    plt.close(fig)
    print(f"  saved {out_path}.pdf / .png")
    return stats_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="scTRP vs scGPT vs XGBoost ablation on one study")
    parser.add_argument("--sctrp_csv", required=True)
    parser.add_argument("--xgboost_csv", required=True)
    parser.add_argument("--scgpt_h5ad", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--alternative", default="greater", choices=["greater", "less", "two-sided"],
                         help="Paired t-test alternative for scTRP-vs-other significance brackets")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scTRP (sample / sample_clone_id source of truth)...")
    sctrp_df = load_sctrp(args.sctrp_csv)
    barcode_map = sctrp_df[["sample", "sample_clone_id"]]

    print("Loading XGBoost...")
    xgboost_df = report_and_drop_unmatched(load_xgboost(args.xgboost_csv, barcode_map), "xgboost")

    print("Loading scGPT...")
    scgpt_df = report_and_drop_unmatched(load_scgpt(args.scgpt_h5ad, barcode_map), "scGPT")

    method_dfs = {"scTRP": sctrp_df, "scGPT": scgpt_df, "xgboost": xgboost_df}

    # ── Cell level ────────────────────────────────────────────────────────────
    print("\n=== Cell-level patient-wise metrics ===")
    cell_rows = []
    for method, df in method_dfs.items():
        pat_df = compute_patient_wise_from_df(df)
        if pat_df.empty:
            print(f"  [skip] {method}: no valid patients")
            continue
        pat_df["method"] = method
        print(f"  {method:10s} — {len(pat_df)} patients | "
              f"mean ROC-AUC={pat_df['ROC_AUC'].mean():.3f}")
        cell_rows.append(pat_df)
    cell_patient_wise = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    cell_patient_wise.to_csv(out_dir / "cell_level_patient_wise.csv", index=False)

    cell_stats = plot_grouped_metric_barplot(
        cell_patient_wise, out_dir / "ablation_3methods_cell_level",
        title="Cell level", alternative=args.alternative,
    )
    pd.DataFrame(cell_stats).to_csv(out_dir / "cell_level_ttest_stats.csv", index=False)

    # ── Clonotype level ──────────────────────────────────────────────────────
    print("\n=== Clonotype-level (max-pooled score) patient-wise metrics ===")
    clono_rows = []
    for method, df in method_dfs.items():
        pooled = pool_clonotype_max(df)
        pat_df = compute_patient_wise_from_df(pooled)
        if pat_df.empty:
            print(f"  [skip] {method}: no valid patients")
            continue
        pat_df["method"] = method
        print(f"  {method:10s} — {len(pat_df)} patients | "
              f"{len(pooled)} clonotypes | mean ROC-AUC={pat_df['ROC_AUC'].mean():.3f}")
        clono_rows.append(pat_df)
    clono_patient_wise = pd.concat(clono_rows, ignore_index=True) if clono_rows else pd.DataFrame()
    clono_patient_wise.to_csv(out_dir / "clonotype_level_patient_wise.csv", index=False)

    clono_stats = plot_grouped_metric_barplot(
        clono_patient_wise, out_dir / "ablation_3methods_clonotype_level",
        title="Clonotype level (max-pooled)", alternative=args.alternative,
    )
    pd.DataFrame(clono_stats).to_csv(out_dir / "clonotype_level_ttest_stats.csv", index=False)

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
