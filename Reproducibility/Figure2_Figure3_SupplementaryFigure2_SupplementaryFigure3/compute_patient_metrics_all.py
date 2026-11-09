#!/usr/bin/env python3
"""
compute_patient_metrics_all.py

Step 3 of unified prediction pipeline.

Loads the raw_cell_scores / pooled_scores CSVs produced by
collect_cell_predictions.py, applies per-patient Jenks thresholding,
computes patient-wise metrics (AUROC, MCC, F1, G-mean, …), then
regenerates the figures from all four analysis notebooks:

  patient_wise_metrics_benchmark.ipynb   → comparison heatmaps / strip+box
  patient_wise_metrics_scTRP.ipynb       → comparison heatmaps / strip+box
  comparison_figure_patient_wise.ipynb   → same, combined
  ablation_scTRP_clonotype_vs_transcription.ipynb → ablation Fig 1-4

One figure set is produced per pool_type (raw, mean, max, median, p75).

Output layout
  {NEW_ROOT}/
    patient_wise/{pool_type}/{method_key}_{study}.csv
    all_patient_wise.csv                   ← all pool_types combined (long format)
    figures/{pool_type}/
      comparison_heatmap_{metric}.pdf
      comparison_stripbox_{metric}.pdf
      ablation_heatmap_{metric}.pdf
      ablation_barplot_{metric}.pdf
      ablation_per_patient_bar_{metric}.pdf
      ablation_per_patient_stripbox_{metric}.pdf

Usage
-----
python compute_patient_metrics_all.py [--in_root DIR] [--pool_types raw,mean,max,median,p75]
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon as _wilcoxon_fn
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

try:
    import jenkspy
    HAS_JENKSPY = True
except ImportError:
    HAS_JENKSPY = False
    print("[warn] jenkspy not found — using median fallback for Jenks breaks")

warnings.filterwarnings("ignore")

# ── Publication rcParams ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":             14,
    "axes.labelsize":        18,
    "axes.titlesize":        18,
    "axes.labelweight":      "bold",
    "axes.titleweight":      "bold",
    "xtick.labelsize":       15,
    "ytick.labelsize":       15,
    "legend.fontsize":       14,
    "legend.title_fontsize": 15,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
    "axes.linewidth":        1.2,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "xtick.major.width":     1.2,
    "ytick.major.width":     1.2,
    "xtick.major.size":      5,
    "ytick.major.size":      5,
    "lines.linewidth":       2.0,
})

# ── Constants ─────────────────────────────────────────────────────────────────
BASE = "/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein"
DEFAULT_IN_ROOT = (
    f"{BASE}/results/classifier/case_study/"
    "ICB_chiffelle_5case_inference_6AllasTrain/unified_cell_predictions"
)

POOL_TYPES  = ["raw", "mean", "max", "median", "p75"]
SIX_STUDIES = ["caushi", "hanada", "lowery", "meng", "oliveira", "zheng"]
ENSEMBLE_STUDIES = ["chiffelle", "moffitt"]
ALL_STUDIES = SIX_STUDIES + ENSEMBLE_STUDIES

METRIC_COLS = ["ROC_AUC", "F1", "Balanced_accuracy", "MCC", "Gmean",
               "Precision", "Recall", "Accuracy", "ARI", "F1_Binary", "Precision_Binary", "Recall_Binary", "n_cells", "n_pos", "n_neg"]

STUDY_ORDER  = ALL_STUDIES
STUDY_LABELS = {
    "caushi":    "Caushi\n(NSCLC, n=4)",
    "hanada":    "Hanada\n(NSCLC, n=4)",
    "lowery":    "Lowery\n(Mel., n=11)",
    "meng":      "Meng\n(PDAC, n=8)",
    "oliveira":  "Oliveira\n(Mel., n=4)",
    "zheng":     "Zheng\n(GI, n=3)",
    "chiffelle": "Chiffelle\n(Mel., n=13)",
    "moffitt": "moffitt\n(NSCLC, n=14)",
}

# Benchmark methods first, then scTRP variants, then XGBoost
BENCH_KEYS = [
    "benchmark_AUCell_Lowery", "benchmark_Seurat_Caushi", "benchmark_Hanada",
    "benchmark_UCell_Meng", "benchmark_UCell_Oliveira", "benchmark_Zheng",
    "benchmark_Manascore", "benchmark_Trace",
]
SCTRP_KEYS  = ["scTRP_simclr", "scTRP_only"]
# Chiffelle-specific: 4 ensemble modes for each of simclr / only
SCTRP_ENSEMBLE_KEYS = [
    # scTRP_simclr
    "scTRP_simclr_score_mean", "scTRP_simclr_rank_sum",
    "scTRP_simclr_vote_fraction", "scTRP_simclr_rank_majority",
    "scTRP_simclr_scoreMean_majority", "scTRP_simclr_rankSum_study", 
    "scTRP_simclr_rankProduct_patient", "scTRP_simclr_rankProduct_study",
    "scTRP_simclr_rankMin_patient", "scTRP_simclr_rankMin_study",
    "scTRP_simclr_zscoreMean_patient", "scTRP_simclr_zscoreMean_study", "scTRP_simclr_zscoreMean_studyPerPatient", 
    # scTRP_only
    "scTRP_only_score_mean",   "scTRP_only_rank_sum",
    "scTRP_only_vote_fraction","scTRP_only_rank_majority",
    "scTRP_only_scoreMean_majority", "scTRP_only_rankSum_study", 
    "scTRP_only_rankProduct_patient", "scTRP_only_rankProduct_study",
    "scTRP_only_rankMin_patient", "scTRP_only_rankMin_study",
    "scTRP_only_zscoreMean_patient", "scTRP_only_zscoreMean_study", "scTRP_only_zscoreMean_studyPerPatient", 
]

        


ABLATION_KEYS   = ["scTRP_simclr", "scTRP_only", "xgboost"]
ALL_METHOD_KEYS = BENCH_KEYS + SCTRP_KEYS + SCTRP_ENSEMBLE_KEYS + ["xgboost"]

METHOD_LABELS = {
    "benchmark_AUCell_Lowery":       "Lowery\n(AUCell)",
    "benchmark_Seurat_Caushi":       "Caushi\n(Seurat)",
    "benchmark_Hanada":              "Hanada",
    "benchmark_UCell_Meng":          "Meng\n(UCell)",
    "benchmark_UCell_Oliveira":      "Oliveira\n(UCell)",
    "benchmark_Zheng":               "Zheng",
    "scTRP_simclr":                  "scTRP\n(+ TCR clonotype)",
    "scTRP_only":                    "scTRP\n(transcription only)",
    
    "scTRP_simclr_score_mean":       "scTRP+TCR\n(score mean)",
    "scTRP_simclr_rank_sum":         "scTRP+TCR\n(rank sum)",
    "scTRP_simclr_vote_fraction":    "scTRP+TCR\n(vote frac.)",
    "scTRP_simclr_rank_majority":    "scTRP+TCR\n(rank+majority)",
    
    "scTRP_simclr_scoreMean_majority": "scTRP+TCR\n(scoreMean+majority)",
    "scTRP_simclr_rankSum_study": "scTRP+TCR\n(rankSum--study)",
    "scTRP_simclr_rankProduct_patient": "scTRP+TCR\n(rankProduct--patient)",
    "scTRP_simclr_rankProduct_study": "scTRP+TCR\n(rankProduct--study)",
    "scTRP_simclr_rankMin_patient": "scTRP+TCR\n(rankMin--patient)",
    "scTRP_simclr_rankMin_study": "scTRP+TCR\n(rankMin--study)",
    "scTRP_simclr_zscoreMean_patient": "scTRP+TCR\n(zscoreMean--patient)",
    "scTRP_simclr_zscoreMean_study": "scTRP+TCR\n(zscoreMean--study)",
    "scTRP_simclr_zscoreMean_studyPerPatient": "scTRP+TCR\n(zscoreMean--studyPerPa)", 
    
    
    "scTRP_only_score_mean":         "scTRP-only\n(score mean)",
    "scTRP_only_rank_sum":           "scTRP-only\n(rank sum)",
    "scTRP_only_vote_fraction":      "scTRP-only\n(vote frac.)",
    "scTRP_only_rank_majority":      "scTRP-only\n(rank+majority)",

    "scTRP_only_scoreMean_majority": "scTRP-only\n(scoreMean+majority)",
    "scTRP_only_rankSum_study": "scTRP-only\n(rankSum--study)",
    "scTRP_only_rankProduct_patient": "scTRP-only\n(rankProduct--patient)",
    "scTRP_only_rankProduct_study": "scTRP-only\n(rankProduct--study)",
    "scTRP_only_rankMin_patient": "scTRP-only\n(rankMin--patient)",
    "scTRP_only_rankMin_study": "scTRP-only\n(rankMin--study)",
    "scTRP_only_zscoreMean_patient": "scTRP-only\n(zscoreMean--patient)",
    "scTRP_only_zscoreMean_study": "scTRP-only\n(zscoreMean--study)",
    "scTRP_only_zscoreMean_studyPerPatient": "scTRP-only\n(zscoreMean--studyPerPa)", 

    
    "xgboost":                       "XGBoost",
    "benchmark_Manascore": "MANAscore",
    "benchmark_Trace": "TRACE",
}

# Color palettes
_bench_colors = sns.color_palette("Blues_d", n_colors=8)[2:]
METHOD_PALETTE = {k: c for k, c in zip(BENCH_KEYS, _bench_colors)}
_c_pal = sns.color_palette("OrRd", 16)[2:]
_o_pal = sns.color_palette("PuBu", 16)[2:]
METHOD_PALETTE["scTRP_simclr"] = _c_pal[0]
METHOD_PALETTE["scTRP_only"]   = _o_pal[0]
METHOD_PALETTE["xgboost"]      = "#888888"
# Ensemble mode sub-keys (lighter shades of the parent color)
for _i, _mode in enumerate(["score_mean", "rank_sum", "vote_fraction", "rank_majority", 
                           "scoreMean_majority", "rankSum_study", "rankProduct_patient", "rankProduct_study",
"rankMin_patient", "rankMin_study", "zscoreMean_patient", "zscoreMean_study", "zscoreMean_studyPerPatient"]):
    _alpha_map = [0.55, 0.70, 0.85, 1.0]
    METHOD_PALETTE[f"scTRP_simclr_{_mode}"] = _c_pal[min(_i, len(_c_pal)-1)]
    METHOD_PALETTE[f"scTRP_only_{_mode}"]   = _o_pal[min(_i, len(_o_pal)-1)]

METRICS_PLOT = [
    ("ROC_AUC", "ROC-AUC",  0.4, 1.0),
    ("MCC",     "MCC",       0.0, 0.65),
    ("F1",      "F1-score",  0.4, 1.0),
    ("Gmean",   "G-mean",    0.4, 1.0),
]

# ── Core metric helpers ───────────────────────────────────────────────────────

def _jenks_thresh(scores):
    if HAS_JENKSPY and len(scores) >= 2:
        try:
            return jenkspy.jenks_breaks(list(scores), n_classes=2)[1]
        except Exception:
            pass
    return float(np.median(scores))


def compute_patient_metrics(labels, preds, scores):
    labels = np.array(labels, dtype=int)
    preds  = np.array(preds,  dtype=int)
    scores = np.array(scores, dtype=float)
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()

    result = {k: np.nan for k in METRIC_COLS}
    result["n_cells"] = len(labels)
    result["n_pos"]   = int(n_pos)
    result["n_neg"]   = int(n_neg)

    try:
        result["Accuracy"]  = accuracy_score(labels, preds)
        result["F1"]        = f1_score(labels, preds, average="weighted", zero_division=0)
        result["Precision"] = precision_score(labels, preds, average="weighted", zero_division=0)
        result["Recall"]    = recall_score(labels, preds, average="weighted", zero_division=0)
        result["F1_Binary"]        = f1_score(labels, preds, average="binary", zero_division=0)
        result["Precision_Binary"] = precision_score(labels, preds, average="binary", zero_division=0)
        result["Recall_Binary"]    = recall_score(labels, preds, average="binary", zero_division=0)
    except Exception:
        pass

    if n_pos >= 1 and n_neg >= 1:
        try:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            result["ROC_AUC"]           = roc_auc_score(labels, scores)
            result["Balanced_accuracy"] = balanced_accuracy_score(labels, preds)
            result["MCC"]               = matthews_corrcoef(labels, preds)
            result["Gmean"]             = (sen * spe) ** 0.5
            result["ARI"]               = adjusted_rand_score(labels, preds)
        except Exception:
            pass

    return result


def compute_patient_wise_from_df(df):
    """
    Apply Jenks per patient on df['score'] to get binary predictions, compute metrics.
    If a 'pred' column is present (e.g., majority-vote ensembles), use it directly
    as binary predictions while still using 'score' for ROC-AUC computation.
    df must have: score, reactivity, sample columns.
    Returns DataFrame[sample, ROC_AUC, MCC, F1, Gmean, ...].
    """
    has_pred_col = "pred" in df.columns
    rows = []
    for samp, grp in df.groupby("sample"):
        g = grp.dropna(subset=["score", "reactivity"])
        if len(g) < 2:
            continue
        scores = g["score"].values.astype(float)
        labels = g["reactivity"].values.astype(int)
        if has_pred_col and g["pred"].notna().all():
            # Use pre-computed binary pred (majority vote / fixed threshold)
            preds = g["pred"].values.astype(int)
        else:
            try:
                thresh = _jenks_thresh(scores)
                preds  = (scores > thresh).astype(int)
            except Exception as e:
                print(f"    skip {samp}: jenks failed ({e})")
                continue
        m = compute_patient_metrics(labels, preds, scores)
        m["sample"] = samp
        rows.append(m)
    return pd.DataFrame(rows, columns=["sample"] + METRIC_COLS) if rows else pd.DataFrame()


# ── Wilcoxon helpers ──────────────────────────────────────────────────────────

def _wilcoxon_stars(a, b, alternative="greater"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == b):
        return np.nan, "ns"
    try:
        _, pval = _wilcoxon_fn(a, b, alternative=alternative)
        star = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        return float(pval), star
    except Exception:
        return np.nan, ""


def _sig_bracket(ax, x1, x2, y, h=0.025, star="", fontsize=13,
                 color="#333333", lw=1.2, show_ns=True):
    if not star or (star == "ns" and not show_ns):
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=lw, color=color, clip_on=False)
    fw = "bold" if star != "ns" else "normal"
    ax.text((x1 + x2) / 2, y + h + 0.003, star,
            ha="center", va="bottom", fontsize=fontsize,
            color=color, fontweight=fw, clip_on=False)


# ── Load all CSVs for one pool_type ──────────────────────────────────────────

def load_cell_csvs(root_dir, pool_type, excludeUnknown=False):
    """
    Load all {method_key}_{study}.csv for given pool_type.
    Returns dict: {(method_key, study): DataFrame}.
    """
    if pool_type == "raw":
        src_dir = Path(root_dir) / "raw_cell_scores"
    else:
        src_dir = Path(root_dir) / "pooled_scores" / pool_type

    result = {}
    for csv_path in sorted(src_dir.glob("*.csv")):
        stem = csv_path.stem
        for study in ALL_STUDIES:
            if stem.endswith(f"_{study}"):
                key = stem[: -(len(study) + 1)]
                df = pd.read_csv(csv_path, index_col=0)
                if excludeUnknown:
                    df['reactivity_fine'] = df['reactivity_fine'].values.astype(int)
                    df = df[df['reactivity_fine'] != 8].copy()
                result[(key, study)] = df
                break
    return result


# ── Compute patient-wise metrics for one pool_type ────────────────────────────

def compute_all_patient_wise(cell_data, pat_dir, pool_type):
    """
    cell_data: dict from load_cell_csvs.
    Saves per-study CSVs and returns long-format DataFrame.
    """
    pat_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for (key, study), df in cell_data.items():
        if "score" not in df.columns or "reactivity" not in df.columns or "sample" not in df.columns:
            print(f"  [skip] {key}/{study}: missing required columns")
            continue

        pat_df = compute_patient_wise_from_df(df)
        if pat_df.empty:
            continue

        pat_df["method"] = key
        pat_df["study"]  = study
        pat_df["pool_type"] = pool_type

        out = pat_dir / f"{key}_{study}.csv"
        pat_df.to_csv(out, index=False)

        n_valid = pat_df["ROC_AUC"].notna().sum()
        print(f"  {key:35s} / {study:10s} — {len(pat_df)} patients "
              f"({n_valid} with AUROC) | mean AUROC = {pat_df['ROC_AUC'].mean():.3f}")
        all_rows.append(pat_df)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting functions
# ═══════════════════════════════════════════════════════════════════════════════

def _method_order(df, preferred_first):
    present = df["method"].unique() if "method" in df.columns else []
    order   = [m for m in preferred_first if m in present]
    extra   = [m for m in present if m not in order]
    return order + extra


def plot_comparison_heatmap(df_all, fig_dir, pool_type):
    """Heatmap: mean patient-wise metrics, method × study."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    method_order = _method_order(df_all, ALL_METHOD_KEYS)
    study_order  = [s for s in STUDY_ORDER if s in df_all["study"].unique()]

    for metric, title, vmin, vmax in METRICS_PLOT:
        if metric not in df_all.columns or df_all[metric].isna().all():
            continue

        pivot = (df_all.groupby(["method", "study"])[metric].mean()
                 .unstack("study")
                 .reindex(index=method_order, columns=study_order))

        nrow, ncol = len(method_order), len(study_order)
        fig, ax = plt.subplots(figsize=(ncol * 1.9 + 1.5, nrow * 1.05 + 0.8),
                               constrained_layout=True)
        im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn_r",
                       vmin=vmin, vmax=vmax, aspect="auto")

        sctrp_idx = method_order.index("scTRP_simclr") if "scTRP_simclr" in method_order else None

        for i in range(nrow):
            for j in range(ncol):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    fw = "bold" if method_order[i] in SCTRP_KEYS else "normal"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=10, color="black", fontweight=fw)
                else:
                    ax.text(j, i, "–", ha="center", va="center",
                            fontsize=9, color="#888888")

        if sctrp_idx is not None:
            ax.axhline(sctrp_idx - 0.5, color="#333", lw=1.5, ls="--")
            ax.axhline(len(BENCH_KEYS) - 0.5, color="#333", lw=0.8, ls=":")

        ax.set_xticks(range(ncol))
        ax.set_xticklabels([STUDY_LABELS.get(s, s) for s in study_order],
                            rotation=0, ha="center", fontsize=11)
        ax.set_yticks(range(nrow))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in method_order], fontsize=11)
        ax.set_title(f"Patient-wise {title}  [{pool_type}]", pad=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
        cbar.set_label(metric, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        base = f"comparison_heatmap_{metric}_{pool_type}"
        fig.savefig(fig_dir / f"{base}.pdf")
        fig.savefig(fig_dir / f"{base}.png", dpi=300)
        plt.close(fig)
        print(f"    saved {base}")


def plot_comparison_stripbox(df_all, fig_dir, pool_type):
    """Strip + box plots: patient-wise metrics, all methods × all studies."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    method_order = _method_order(df_all, ALL_METHOD_KEYS)
    study_order  = [s for s in STUDY_ORDER if s in df_all["study"].unique()]
    y_pos = {m: i for i, m in enumerate(method_order)}
    n_meth = len(method_order)
    rng = np.random.RandomState(42)

    for metric, ylabel, xmin, xmax in METRICS_PLOT:
        if metric not in df_all.columns or df_all[metric].isna().all():
            continue

        ncols = len(study_order)
        fig, axes = plt.subplots(
            1, ncols,
            figsize=(ncols * 2.5, n_meth * 0.75 + 1.8),
            sharey=True, constrained_layout=True,
        )
        if ncols == 1:
            axes = [axes]

        for ax, study in zip(axes, study_order):
            sub = df_all[df_all["study"] == study]
            sctrp_df = (sub[sub["method"] == "scTRP_simclr"][["sample", metric]]
                        .dropna().set_index("sample"))

            for method in method_order:
                msub  = sub[sub["method"] == method]
                yvals = msub[metric].dropna().values
                if len(yvals) == 0:
                    continue
                yp    = y_pos[method]
                color = METHOD_PALETTE.get(method, "#aaa")
                q1, q2, q3 = np.percentile(yvals, [25, 50, 75])
                iqr = q3 - q1
                lo  = max(yvals.min(), q1 - 1.5 * iqr)
                hi  = min(yvals.max(), q3 + 1.5 * iqr)
                ax.barh(yp, q3 - q1, left=q1, height=0.50,
                        color=color, alpha=0.35, zorder=2)
                ax.vlines(q2, yp - 0.25, yp + 0.25, color=color, lw=2.5, zorder=4)
                ax.hlines(yp, lo, hi, color=color, lw=1.2, zorder=3)
                jit = rng.uniform(-0.18, 0.18, size=len(yvals))
                ax.scatter(yvals, yp + jit, color=color, s=18, alpha=0.75,
                           zorder=5, edgecolors="none")

            # Paired Wilcoxon (scTRP_simclr > each benchmark)
            annot_x = xmax + (xmax - xmin) * 0.04
            for bench_key in BENCH_KEYS:
                msub   = sub[sub["method"] == bench_key]
                bench_df = (msub[["sample", metric]].dropna().set_index("sample"))
                common = sctrp_df.index.intersection(bench_df.index)
                if len(common) < 3:
                    continue
                a = sctrp_df.loc[common, metric].values.astype(float)
                b = bench_df.loc[common, metric].values.astype(float)
                _, star = _wilcoxon_stars(a, b)
                if star and star != "ns":
                    yp_bench = y_pos[bench_key]
                    yp_s     = y_pos["scTRP_simclr"]
                    yc       = (yp_bench + yp_s) / 2
                    ax.text(annot_x, yc, star, ha="left", va="center",
                            fontsize=9, color="#333", fontweight="bold")
                    annot_x += (xmax - xmin) * 0.09

            ax.set_xlim(xmin - 0.02, xmax + (xmax - xmin) * 0.35)
            ax.set_title(STUDY_LABELS.get(study, study), fontsize=11, pad=3)
            ax.axvline(0.5, color="lightgray", lw=0.8, ls="--", zorder=1)

        axes[0].set_yticks(range(n_meth))
        axes[0].set_yticklabels([METHOD_LABELS.get(m, m) for m in method_order],
                                  fontsize=9)
        axes[0].set_ylabel(ylabel, fontsize=14)

        fig.suptitle(f"Patient-wise {ylabel}  [{pool_type}]", y=1.01,
                     fontsize=14, fontweight="bold")

        base = f"comparison_stripbox_{metric}_{pool_type}"
        fig.savefig(fig_dir / f"{base}.pdf")
        fig.savefig(fig_dir / f"{base}.png", dpi=300)
        plt.close(fig)
        print(f"    saved {base}")


def plot_ablation_heatmap(df_all, fig_dir, pool_type):
    """Heatmap for ablation trio: scTRP_simclr, scTRP_only, xgboost."""
    abl_methods = [m for m in ABLATION_KEYS if m in df_all["method"].unique()]
    if not abl_methods:
        return
    df_sub = df_all[df_all["method"].isin(abl_methods)]
    study_order = [s for s in STUDY_ORDER if s in df_sub["study"].unique()]

    for metric, title, vmin, vmax in METRICS_PLOT:
        if metric not in df_sub.columns:
            continue
        pivot = (df_sub.groupby(["method", "study"])[metric].mean()
                 .unstack("study")
                 .reindex(index=abl_methods, columns=study_order))
        nrow, ncol = len(abl_methods), len(study_order)
        fig, ax = plt.subplots(figsize=(ncol * 2.1 + 1.5, nrow * 1.35 + 1.0),
                               constrained_layout=True)
        im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn_r",
                       vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(nrow):
            for j in range(ncol):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    fw = "bold" if abl_methods[i] == "scTRP_simclr" else "normal"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=13, color="black", fontweight=fw)
        ax.set_xticks(range(ncol))
        ax.set_xticklabels([STUDY_LABELS.get(s, s) for s in study_order],
                            ha="center", fontsize=13)
        ax.set_yticks(range(nrow))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in abl_methods], fontsize=13)
        ax.set_title(f"Ablation — {title}  [{pool_type}]", pad=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
        cbar.set_label(metric, fontsize=13)
        cbar.ax.tick_params(labelsize=12)
        base = f"ablation_heatmap_{metric}_{pool_type}"
        fig.savefig(fig_dir / f"{base}.pdf")
        fig.savefig(fig_dir / f"{base}.png", dpi=300)
        plt.close(fig)
        print(f"    saved {base}")


def plot_ablation_barplot(df_all, fig_dir, pool_type):
    """
    Grouped bar plot: mean patient-wise metric per study for ablation methods.
    Per-patient scatter overlaid + Wilcoxon bracket (scTRP_simclr > scTRP_only).
    """
    abl_methods = [m for m in ABLATION_KEYS if m in df_all["method"].unique()]
    if not abl_methods:
        return

    df_sub = df_all[df_all["method"].isin(abl_methods)]
    study_6     = [s for s in SIX_STUDIES if s in df_sub["study"].unique()]
    #study_chiff = ["chiffelle"] if "chiffelle" in df_sub["study"].unique() else []
    study_chiff = ENSEMBLE_STUDIES if "moffitt" in df_sub["study"].unique() else ["chiffelle"]

    n_meth = len(abl_methods)
    bar_w  = 0.80 / n_meth
    rng    = np.random.RandomState(42)

    key_s = "scTRP_simclr"
    key_o = "scTRP_only"
    idx_s = abl_methods.index(key_s) if key_s in abl_methods else -1
    idx_o = abl_methods.index(key_o) if key_o in abl_methods else -1

    for metric, metric_label, vmin, vmax in METRICS_PLOT:
        if metric not in df_sub.columns:
            continue

        fig, axes = plt.subplots(
            1, 2,
            figsize=(max(len(study_6), 1) * 1.9 + 5.5, 7.2),
            gridspec_kw={"width_ratios": [max(len(study_6), 1), 1]},
            sharey=True, constrained_layout=True,
        )

        for ax, studies_sub, panel_title in zip(
                axes,
                [study_6, study_chiff],
                ["6 leave-one-out test studies", "Independent test sets"]):
            if not studies_sub:
                ax.set_visible(False)
                continue

            x = np.arange(len(studies_sub))
            y_study_max = {s: vmin for s in studies_sub}

            # ── Bars ──────────────────────────────────────────────────────────
            for i, method in enumerate(abl_methods):
                if method == "xgboost" and studies_sub != study_chiff:
                    continue
                offset = (i - n_meth / 2 + 0.5) * bar_w
                vals = [df_sub[(df_sub["study"] == s) &
                               (df_sub["method"] == method)][metric].mean()
                        for s in studies_sub]
                bars = ax.bar(
                    x + offset, vals, width=bar_w * 0.88,
                    color=METHOD_PALETTE.get(method, "#aaa"), alpha=0.72,
                    edgecolor="white", linewidth=0.5,
                    label=METHOD_LABELS.get(method, method), zorder=3,
                )
                for bar, v in zip(bars, vals):
                    if not np.isnan(v) and v > vmin:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                v + 0.008, f"{v:.3f}",
                                ha="center", va="bottom",
                                fontsize=12, rotation=90, fontweight="bold",
                                color="#111111", zorder=7)

            # ── Per-patient scatter ───────────────────────────────────────────
            pat_by_method = {m: df_sub[df_sub["method"] == m].set_index("sample")
                             for m in abl_methods}

            for i, method in enumerate(abl_methods):
                if method == "xgboost" and studies_sub != study_chiff:
                    continue
                offset = (i - n_meth / 2 + 0.5) * bar_w
                for j, study in enumerate(studies_sub):
                    pat_s = pat_by_method[method]
                    yvals = pat_s[pat_s["study"] == study][metric].dropna().values
                    if len(yvals) == 0:
                        continue
                    jitter = rng.uniform(-bar_w * 0.27, bar_w * 0.27, size=len(yvals))
                    ax.scatter(
                        j + offset + jitter, yvals,
                        s=28, alpha=0.85, zorder=6,
                        color=METHOD_PALETTE.get(method, "#aaa"),
                        edgecolors="white", linewidths=0.5,
                    )
                    y_study_max[study] = max(y_study_max[study], float(np.nanmax(yvals)))

            # ── Wilcoxon brackets (scTRP_simclr > scTRP_only) ────────────────
            if idx_s >= 0 and idx_o >= 0:
                pat_s_all = pat_by_method.get(key_s, pd.DataFrame())
                pat_o_all = pat_by_method.get(key_o, pd.DataFrame())
                for j, study in enumerate(studies_sub):
                    ps_i = pat_s_all[pat_s_all["study"] == study] if "study" in pat_s_all.columns else pat_s_all
                    po_i = pat_o_all[pat_o_all["study"] == study] if "study" in pat_o_all.columns else pat_o_all
                    common = ps_i.index.intersection(po_i.index)
                    if len(common) < 3:
                        continue
                    a = ps_i.loc[common, metric].values.astype(float)
                    b = po_i.loc[common, metric].values.astype(float)
                    _, star = _wilcoxon_stars(a, b)
                    x_s = j + (idx_s - n_meth / 2 + 0.5) * bar_w
                    x_o = j + (idx_o - n_meth / 2 + 0.5) * bar_w
                    y_base = max(y_study_max[study], vmin + 0.02) + 0.04
                    _sig_bracket(ax, min(x_s, x_o), max(x_s, x_o),
                                 y_base, h=0.028, star=star, show_ns=False)

            ax.set_xticks(x)
            ax.set_xticklabels([STUDY_LABELS.get(s, s) for s in studies_sub],
                                ha="center", fontsize=12)
            ax.set_title(panel_title, fontsize=13, pad=5)
            ax.axhline(0.5, color="lightgray", lw=0.8, ls="--", zorder=1)
            ax.set_ylim(vmin - 0.04, vmax + 0.12)

        axes[0].set_ylabel(metric_label, fontsize=15)
        handles = [plt.Rectangle((0, 0), 1, 1,
                                  color=METHOD_PALETTE.get(m, "#aaa"), alpha=0.72)
                   for m in abl_methods]
        labels  = [METHOD_LABELS.get(m, m).replace("\n", " ") for m in abl_methods]
        axes[0].legend(handles, labels, loc="lower left", frameon=False, fontsize=11)
        fig.suptitle(f"Ablation: {metric_label}  [{pool_type}]", y=1.01,
                     fontsize=14, fontweight="bold")

        base = f"ablation_barplot_{metric}_{pool_type}"
        fig.savefig(fig_dir / f"{base}.pdf")
        fig.savefig(fig_dir / f"{base}.png", dpi=300)
        plt.close(fig)
        print(f"    saved {base}")


def plot_ablation_per_patient_stripbox(df_all, fig_dir, pool_type):
    """
    Strip + box plots per patient: ALL datasets, ablation methods only.
    Wilcoxon bracket (scTRP_simclr > scTRP_only) per study.
    """
    abl_methods = [m for m in ABLATION_KEYS if m in df_all["method"].unique()]
    if not abl_methods:
        return

    df_sub = df_all[df_all["method"].isin(abl_methods)]
    study_order = [s for s in STUDY_ORDER if s in df_sub["study"].unique()]
    y_pos  = {m: i for i, m in enumerate(abl_methods)}
    n_meth = len(abl_methods)
    rng    = np.random.RandomState(42)

    key_s = "scTRP_simclr"
    key_o = "scTRP_only"

    for metric, ylabel, xmin, xmax in METRICS_PLOT:
        if metric not in df_sub.columns or df_sub[metric].isna().all():
            continue

        ncols = len(study_order)
        fig, axes = plt.subplots(
            1, ncols,
            figsize=(ncols * 2.5, n_meth * 0.85 + 2.0),
            sharey=True, constrained_layout=True,
        )
        if ncols == 1:
            axes = [axes]

        pat_by_method = {m: df_sub[df_sub["method"] == m] for m in abl_methods}

        for ax, study in zip(axes, study_order):
            for method in abl_methods:
                msub  = pat_by_method[method]
                msub  = msub[msub["study"] == study]
                yvals = msub[metric].dropna().values
                if len(yvals) == 0:
                    continue
                yp    = y_pos[method]
                color = METHOD_PALETTE.get(method, "#aaa")
                q1, q2, q3 = np.percentile(yvals, [25, 50, 75])
                iqr = q3 - q1
                lo  = max(yvals.min(), q1 - 1.5 * iqr)
                hi  = min(yvals.max(), q3 + 1.5 * iqr)
                ax.barh(yp, q3 - q1, left=q1, height=0.55,
                        color=color, alpha=0.35, zorder=2)
                ax.vlines(q2, yp - 0.275, yp + 0.275, color=color, lw=2.5, zorder=4)
                ax.hlines(yp, lo, hi, color=color, lw=1.2, zorder=3)
                jit = rng.uniform(-0.20, 0.20, size=len(yvals))
                ax.scatter(yvals, yp + jit, color=color, s=22, alpha=0.80,
                           zorder=5, edgecolors="white", linewidths=0.4)

            # Wilcoxon between scTRP_simclr and scTRP_only
            ps = pat_by_method.get(key_s, pd.DataFrame())
            po = pat_by_method.get(key_o, pd.DataFrame())
            ps_s = ps[ps["study"] == study].dropna(subset=[metric]).set_index("sample")
            po_s = po[po["study"] == study].dropna(subset=[metric]).set_index("sample")
            common = ps_s.index.intersection(po_s.index)
            if len(common) >= 3:
                a = ps_s.loc[common, metric].values.astype(float)
                b = po_s.loc[common, metric].values.astype(float)
                _, star = _wilcoxon_stars(a, b)
                if star and star != "ns":
                    x_s = y_pos[key_s]
                    x_o = y_pos[key_o]
                    xc  = (x_s + x_o) / 2
                    annot_x = xmax + (xmax - xmin) * 0.04
                    ax.text(annot_x, xc, star, ha="left", va="center",
                            fontsize=11, color="#d62728", fontweight="bold")

            ax.set_xlim(xmin - 0.02, xmax + (xmax - xmin) * 0.25)
            ax.set_title(STUDY_LABELS.get(study, study), fontsize=11, pad=3)
            ax.axvline(0.5, color="lightgray", lw=0.8, ls="--", zorder=1)

        axes[0].set_yticks(range(n_meth))
        axes[0].set_yticklabels([METHOD_LABELS.get(m, m) for m in abl_methods], fontsize=10)
        axes[0].set_ylabel(ylabel, fontsize=14)
        fig.suptitle(f"Ablation per-patient: {ylabel}  [{pool_type}]", y=1.01,
                     fontsize=14, fontweight="bold")

        base = f"ablation_per_patient_stripbox_{metric}_{pool_type}"
        fig.savefig(fig_dir / f"{base}.pdf")
        fig.savefig(fig_dir / f"{base}.png", dpi=300)
        plt.close(fig)
        print(f"    saved {base}")


def plot_delta_heatmap(df_all, fig_dir, pool_type):
    """ΔMetric heatmap: scTRP_simclr minus scTRP_only per study."""
    df_s = df_all[df_all["method"] == "scTRP_simclr"]
    df_o = df_all[df_all["method"] == "scTRP_only"]
    if df_s.empty or df_o.empty:
        return

    study_order = [s for s in STUDY_ORDER if s in df_all["study"].unique()]
    metrics = [m for m, *_ in METRICS_PLOT]

    delta_vals = {}
    for metric in metrics:
        row = {}
        for study in study_order:
            mean_s = df_s[df_s["study"] == study][metric].mean()
            mean_o = df_o[df_o["study"] == study][metric].mean()
            row[study] = mean_s - mean_o if not (np.isnan(mean_s) or np.isnan(mean_o)) else np.nan
        delta_vals[metric] = row
    delta_df = pd.DataFrame(delta_vals, index=study_order).T

    nrow, ncol = len(metrics), len(study_order)
    fig, ax = plt.subplots(figsize=(ncol * 1.9 + 1.5, nrow * 1.05 + 0.8),
                           constrained_layout=True)
    lim = np.nanmax(np.abs(delta_df.values)) + 0.02
    im = ax.imshow(delta_df.values.astype(float), cmap="RdBu_r",
                   vmin=-lim, vmax=lim, aspect="auto")
    for i in range(nrow):
        for j in range(ncol):
            val = delta_df.values[i, j]
            if not np.isnan(val):
                color = "darkred" if val > 0 else "navy"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([STUDY_LABELS.get(s, s) for s in study_order],
                        ha="center", fontsize=11)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([m for m in metrics], fontsize=11)
    ax.set_title(f"Δ metric (scTRP+TCR − transcription only)  [{pool_type}]", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.8, label="Δ")
    base = f"ablation_delta_heatmap_{pool_type}"
    fig.savefig(fig_dir / f"{base}.pdf")
    fig.savefig(fig_dir / f"{base}.png", dpi=300)
    plt.close(fig)
    print(f"    saved {base}")


# ── Precision@k helpers ───────────────────────────────────────────────────────

def _topn_precision(scores, labels, k):
    """Fraction of reactive cells/clones in the top-k by score."""
    order = np.argsort(scores)[::-1]
    actual_k = min(k, len(order))
    return float(np.array(labels)[order[:actual_k]].mean()) if actual_k > 0 else np.nan


def _per_patient_topk_series(df, k, pat_col="sample"):
    """Per-patient precision@k; returns Series indexed by patient."""
    rows = {}
    for pat, grp in df.groupby(pat_col):
        g = grp.dropna(subset=["score", "reactivity"])
        if len(g) == 0:
            continue
        rows[pat] = _topn_precision(
            g["score"].values.astype(float),
            g["reactivity"].values.astype(int),
            k
        )
    return pd.Series(rows)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from collections import defaultdict
from scipy.stats import wilcoxon as _wtest

# Plot 1: study-level ensemble modes only (valid on a global axis)
STUDY_LEVEL_ENSEMBLE_KEYS = [
    "scTRP_simclr_score_mean", "scTRP_simclr_scoreMean_majority", "scTRP_simclr_rankSum_study",
    "scTRP_simclr_rankProduct_study", "scTRP_simclr_rankMin_study", "scTRP_simclr_zscoreMean_study", "scTRP_simclr_zscoreMean_studyPerPatient",
    "scTRP_only_score_mean", "scTRP_only_scoreMean_majority", "scTRP_only_rankSum_study",
    "scTRP_only_rankProduct_study", "scTRP_only_rankMin_study", "scTRP_only_zscoreMean_study", "scTRP_only_zscoreMean_studyPerPatient",
]

# Plot 2: all ensemble modes (per-patient top-k is valid for patient-relative scores too)
ALL_ENSEMBLE_KEYS = [
    "scTRP_simclr_score_mean", "scTRP_simclr_rank_sum", "scTRP_simclr_rank_majority",
    "scTRP_simclr_scoreMean_majority", "scTRP_simclr_rankSum_study",
    "scTRP_simclr_rankProduct_patient", "scTRP_simclr_rankProduct_study",
    "scTRP_simclr_rankMin_patient", "scTRP_simclr_rankMin_study",
    "scTRP_simclr_zscoreMean_patient", "scTRP_simclr_zscoreMean_study", "scTRP_simclr_zscoreMean_studyPerPatient",
    "scTRP_only_score_mean", "scTRP_only_rank_sum", "scTRP_only_rank_majority",
    "scTRP_only_scoreMean_majority", "scTRP_only_rankSum_study",
    "scTRP_only_rankProduct_patient", "scTRP_only_rankProduct_study",
    "scTRP_only_rankMin_patient", "scTRP_only_rankMin_study",
    "scTRP_only_zscoreMean_patient", "scTRP_only_zscoreMean_study", "scTRP_only_zscoreMean_studyPerPatient",
]


def _dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def _build_method_order(available, ensemble_keys):
    """benchmarks (incl. xgboost) → scTRP base → ensemble modes; only keys present."""
    bench = [k for k in (list(BENCH_KEYS) + ["xgboost"]) if k in available]
    base  = [k for k in SCTRP_KEYS if k in available]
    ens   = [k for k in ensemble_keys if k in available]
    return _dedup_keep_order(bench + base + ens)


def _method_colors(method_order):
    """Known keys keep METHOD_PALETTE; unknown ensemble modes get distinct tab20 colors."""
    fallback = cm.get_cmap("tab20")
    colors, fi = {}, 0
    for m in method_order:
        if m in METHOD_PALETTE:
            colors[m] = METHOD_PALETTE[m]
        else:
            colors[m] = fallback(fi % 20); fi += 1
    return colors


def _prep_method_df(df_m, pool_type):
    """Non-mutating: copy → dedup clonotypes for pooled types → drop NaNs.

    (The original did drop_duplicates(inplace=True) on the cached cell_data df,
    which silently mutated the shared object across pool_types — fixed here.)
    """
    d = df_m.copy()
    if pool_type != "raw" and "sample_clone_id" in d.columns:
        d = d.drop_duplicates(subset="sample_clone_id")
    return d.dropna(subset=["score", "reactivity"])


def _star_from_p(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns")


def _paired_topk_stats(df_ref, df_m, k, alternative, min_pairs=3):
    """
    Full per-patient paired Wilcoxon result for precision@k (ref vs method).
    Patients are the paired replicates. Returns statistic, p, star, n, and the
    per-patient means so a two-sided test's direction is still recoverable.
    """
    p_ref = _per_patient_topk_series(df_ref, k)
    p_m   = _per_patient_topk_series(df_m,   k)
    common = p_ref.index.intersection(p_m.index)
    n = len(common)
    out = {"k": k, "n_patients": n, "statistic": np.nan, "pval": np.nan,
           "star": "", "ref_mean": np.nan, "method_mean": np.nan, "mean_diff": np.nan}
    if n == 0:
        return out
    a = p_ref.loc[common].values.astype(float)
    b = p_m.loc[common].values.astype(float)
    out["ref_mean"], out["method_mean"] = float(a.mean()), float(b.mean())
    out["mean_diff"] = float((a - b).mean())          # ref − method
    if n < min_pairs:
        return out
    if np.all(a - b == 0):
        out["star"] = "ns"; return out
    try:
        stat, pv = _wtest(a, b, alternative=alternative)
        out["statistic"], out["pval"], out["star"] = float(stat), float(pv), _star_from_p(pv)
    except Exception:
        pass
    return out


def _paired_topk_star(df_ref, df_m, k, alternative, min_pairs=3):
    """Star only — thin wrapper so plotting and CSV share one code path."""
    return _paired_topk_stats(df_ref, df_m, k, alternative, min_pairs)["star"]


def _build_perpatient_long(prepped, method_order, k_vals):
    """Tidy long frame: one row per (method, k, patient) precision@k value."""
    rows = []
    for m in method_order:
        for k in k_vals:
            s = _per_patient_topk_series(prepped[m], k)
            for patient, val in s.items():
                rows.append({"method": m, "k": int(k), "patient": patient,
                             "precision": float(val)})
    return pd.DataFrame(rows)


def _build_stats_long(prepped, method_order, ref_key, k_vals, alternative):
    """Tidy long frame: ref-vs-method paired Wilcoxon at each k."""
    if ref_key is None:
        return pd.DataFrame()
    df_ref = prepped[ref_key]
    rows = [{"ref": ref_key, "method": m, "alternative": alternative,
             **_paired_topk_stats(df_ref, prepped[m], k, alternative)}
            for m in method_order if m != ref_key for k in k_vals]
    return pd.DataFrame(rows)


def plot_study_precision_bar(cell_data, fig_dir, pool_type,
                             k_vals=(10, 20, 30, 40, 50),
                             ensemble_keys=STUDY_LEVEL_ENSEMBLE_KEYS,
                             ref_key_pref=("scTRP_simclr_score_mean", "scTRP_simclr"),
                             alternative="greater", removeNoPositivePatient=False, study_="chiffelle"):
    """
    Plot 1 — Grouped bar: study-level precision@k on input study_ for every method.
    Bar value = global top-k precision over the whole study.
    Stars = per-patient paired Wilcoxon (ref vs that method); patients are replicates.
    NOTE: bar value is study-level but the test is per-patient, so a bar can be
    tall yet still lose the paired test, and vice versa — this matches the
    original figure's design.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    chif = {key: df for (key, study), df in cell_data.items() if study == study_}
    tem_sufix = ''
    if removeNoPositivePatient:
        # derive the retained patient set ONCE from ground truth, apply to all methods,
        # so every method is evaluated on an identical patient set (paired tests stay aligned)
        tem_sufix = '_rmNopositivePatient'
        _ref = next(iter(chif.values()))
        _nuniq = _ref.groupby("sample")["reactivity"].nunique()
        keep_samples = set(_nuniq[_nuniq > 1].index)
        dropped = sorted(set(_nuniq.index) - keep_samples)
        if dropped:
            print(f"    [filter] dropping {len(dropped)} single-class patient(s): {dropped}")
        chif = {key: df[df["sample"].isin(keep_samples)].copy()
                for key, df in chif.items()}
    if not chif:
        print(f"    [skip] no {study_} data for pool_type={pool_type}"); return

    method_order = _build_method_order(chif, ensemble_keys)
    if not method_order:
        print("    [skip] no recognisable methods"); return
    colors = _method_colors(method_order)


    ref_key = next((k for k in ref_key_pref if k in chif), None)
    prepped = {m: _prep_method_df(chif[m], pool_type) for m in method_order}

    _any = next(iter(prepped.values()))
    baseline = _any["reactivity"].astype(int).mean() if "reactivity" in _any else np.nan

    # study-level precision@k
    bar_data = pd.DataFrame(
        {m: {k: _topn_precision(g["score"].values.astype(float),
                                g["reactivity"].values.astype(int), k)
             for k in k_vals}
         for m, g in prepped.items()}
    ).T.reindex(method_order)

    # per-patient stars: ref vs each other method
    stars = defaultdict(dict)
    if ref_key:
        df_ref = prepped[ref_key]
        for m in method_order:
            if m == ref_key:
                continue
            for k in k_vals:
                stars[m][k] = _paired_topk_star(df_ref, prepped[m], k, alternative)

    # ── Save values + statistics ─────────────────────────────────────────────
    pp_long = _build_perpatient_long(prepped, method_order, k_vals)
    pp_long.insert(0, "pool_type", pool_type)
    pp_long.to_csv(fig_dir / f"fig2_study_precision_{pool_type}_perpatient_values{tem_sufix}.csv",
                   index=False)

    # study-level bar values (the numbers the bars actually show)
    study_long = (bar_data.reset_index().rename(columns={"index": "method"})
                  .melt(id_vars="method", var_name="k", value_name="study_precision"))
    study_long["k"] = study_long["k"].astype(int)
    study_long.insert(0, "pool_type", pool_type)
    study_long.to_csv(fig_dir / f"fig2_study_precision_{pool_type}_studylevel_values{tem_sufix}.csv",
                      index=False)

    stats_long = _build_stats_long(prepped, method_order, ref_key, k_vals, alternative)
    if not stats_long.empty:
        stats_long.insert(0, "pool_type", pool_type)
        stats_long.to_csv(fig_dir / f"fig2_study_precision_{pool_type}_stats{tem_sufix}.csv",
                          index=False)
    
    n_m, n_k = len(method_order), len(k_vals)
    x = np.arange(n_k); bar_w = 0.80 / n_m
    fig, ax = plt.subplots(figsize=(max(11, n_m * n_k * 0.16 + 3), 6),
                           constrained_layout=True)

    for i, m in enumerate(method_order):
        offset = (i - n_m / 2 + 0.5) * bar_w
        vals   = [bar_data.loc[m, k] for k in k_vals]
        is_ref = (m == ref_key)
        bars = ax.bar(x + offset, vals, width=bar_w * 0.92, color=colors[m],
                      edgecolor=("black" if is_ref else "white"),
                      linewidth=(1.1 if is_ref else 0.4),
                      label=METHOD_LABELS.get(m, m).replace("\n", " "), zorder=3)
        for bar, v, k in zip(bars, vals, k_vals):
            if np.isnan(v):
                continue
            s = "" if is_ref else stars[m].get(k, "")
            if s and s != "ns":
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006, s,
                        ha="center", va="bottom", fontsize=10, color=colors[m])

    if not np.isnan(baseline):
        ax.axhline(baseline, color="#888", lw=1.1, ls="--", zorder=1,
                   label=f"Baseline ({baseline:.0%})")

    ax.set_xticks(x); ax.set_xticklabels([f"Top-{k}" for k in k_vals])
    ax.set_ylabel("Precision (fraction reactive)")
    ax.set_title(f"Study-level precision@k — {study_}  [{pool_type}] {tem_sufix}",
                 pad=8, fontweight="bold")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.grid(True, lw=0.5, color="#eee", zorder=0)
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=3)
    ax.annotate(f"Stars: per-patient paired Wilcoxon ({ref_key} {'>' if alternative=='greater' else '≠'} method); "
                f"* p<0.05  ** p<0.01  *** p<0.001. Black edge = reference.",
                xy=(0.0, -0.13), xycoords="axes fraction", fontsize=12, color="gray")

    base = f"fig2_study_precision_bar_{pool_type}{tem_sufix}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(fig_dir / f"{base}.{ext}", dpi=300)
    plt.close(fig); print(f"    saved {base}")


def plot_perpatient_precision_box(cell_data, fig_dir, pool_type,
                                  k_vals=(10, 20, 30, 40, 50),
                                  ensemble_keys=ALL_ENSEMBLE_KEYS,
                                  ref_key_pref=("scTRP_simclr_score_mean", "scTRP_simclr"),
                                  alternative="greater", removeNoPositivePatient=False, study_="chiffelle"):
    """
    Plot 2 — Box + jitter of per-patient precision@k, one figure per k.
    Each point = one patient's top-k precision (within that patient).
    Value shown and the stat test use the SAME per-patient series, so they are
    fully consistent here (unlike Plot 1). Stars: ref vs method paired Wilcoxon.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    chif = {key: df for (key, study), df in cell_data.items() if study == study_}
    if not chif:
        print(f"    [skip] no {study_} data for pool_type={pool_type}"); return
    tem_sufix = ''
    if removeNoPositivePatient:
        # derive the retained patient set ONCE from ground truth, apply to all methods,
        # so every method is evaluated on an identical patient set (paired tests stay aligned)
        tem_sufix = '_rmNopositivePatient'
        _ref = next(iter(chif.values()))
        _nuniq = _ref.groupby("sample")["reactivity"].nunique()
        keep_samples = set(_nuniq[_nuniq > 1].index)
        dropped = sorted(set(_nuniq.index) - keep_samples)
        if dropped:
            print(f"    [filter] dropping {len(dropped)} single-class patient(s): {dropped}")
        chif = {key: df[df["sample"].isin(keep_samples)].copy()
                for key, df in chif.items()}
    
    method_order = _build_method_order(chif, ensemble_keys)
    if not method_order:
        print("    [skip] no recognisable methods"); return
    colors  = _method_colors(method_order)
    ref_key = next((k for k in ref_key_pref if k in chif), None)
    prepped = {m: _prep_method_df(chif[m], pool_type) for m in method_order}
    labels  = [METHOD_LABELS.get(m, m).replace("\n", " ") for m in method_order]

    _any = next(iter(prepped.values()))
    baseline = _any["reactivity"].astype(int).mean() if "reactivity" in _any else np.nan

    # ── Save values + statistics (all k, once) ───────────────────────────────
    pp_long = _build_perpatient_long(prepped, method_order, k_vals)
    pp_long.insert(0, "pool_type", pool_type)
    pp_long.to_csv(fig_dir / f"fig3_perpatient_precision_{pool_type}_values{tem_sufix}.csv",
                   index=False)

    stats_long = _build_stats_long(prepped, method_order, ref_key, k_vals, alternative)
    if not stats_long.empty:
        stats_long.insert(0, "pool_type", pool_type)
        stats_long.to_csv(fig_dir / f"fig3_perpatient_precision_{pool_type}_stats{tem_sufix}.csv",
                          index=False)

    rng = np.random.default_rng(0)
    for k in k_vals:
        series = {m: _per_patient_topk_series(prepped[m], k) for m in method_order}
        data   = [series[m].values.astype(float) for m in method_order]

        fig, ax = plt.subplots(figsize=(max(10, len(method_order) * 0.5 + 2), 6),
                               constrained_layout=True)
        pos = np.arange(1, len(method_order) + 1)
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.2))
        for patch, m in zip(bp["boxes"], method_order):
            patch.set_facecolor(colors[m]); patch.set_alpha(0.55)

        for p, vals, m in zip(pos, data, method_order):
            jit = rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(p + jit, vals, s=9, color=colors[m], alpha=0.8,
                       edgecolor="white", linewidth=0.2, zorder=3)

        # stars above each non-ref box
        if ref_key:
            df_ref = prepped[ref_key]
            ytop = np.nanmax([np.nanmax(v) if len(v) else np.nan for v in data])
            for p, m, vals in zip(pos, method_order, data):
                if m == ref_key or not len(vals):
                    continue
                s = _paired_topk_star(df_ref, prepped[m], k, alternative)
                if s and s != "ns":
                    ax.text(p, np.nanmax(vals) + 0.02, s, ha="center",
                            va="bottom", fontsize=12, color=colors[m])
            if ref_key in method_order:  # mark the reference box
                rp = pos[method_order.index(ref_key)]
                ax.text(rp, -0.02, "ref", ha="center", va="top",
                        fontsize=7, color="black", transform=ax.get_xaxis_transform())

        if not np.isnan(baseline):
            ax.axhline(baseline, color="#888", lw=1.0, ls="--",
                       label=f"Reactive fraction ({baseline:.0%})")
            ax.legend(frameon=False, fontsize=8, loc="upper right")

        ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(f"Per-patient precision@{k}")
        ax.set_ylim(-0.02, 1.03)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"Per-patient precision@{k} — {study_}  [{pool_type}] {tem_sufix}",
                     pad=8, fontweight="bold")
        ax.yaxis.grid(True, lw=0.5, color="#eee")
        ax.annotate(f"Each point = one patient. Stars: paired Wilcoxon "
                    f"({ref_key} {'>' if alternative=='greater' else '≠'} method).",
                    xy=(0.0, -0.20), xycoords="axes fraction", fontsize=12, color="gray")

        base = f"fig3_perpatient_precision_box_k{k}_{pool_type}{tem_sufix}"
        for ext in ("pdf", "png", "svg"):
            fig.savefig(fig_dir / f"{base}.{ext}", dpi=300)
        plt.close(fig); print(f"    saved {base}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compute patient-wise metrics + generate all plots")
    parser.add_argument("--in_root",    default=DEFAULT_IN_ROOT,
                        help="Root dir from collect_cell_predictions.py")
    parser.add_argument("--pool_types", default=",".join(POOL_TYPES),
                        help="Comma-separated pool types to process")
    parser.add_argument("--no_plots",   action="store_true",
                        help="Skip figure generation (metrics only)")
    parser.add_argument("--no_unknownCells",   action="store_true",
                        help="Exclude cells with unknown reactivity")
    args = parser.parse_args()

    in_root     = Path(args.in_root)
    pool_types  = [p.strip() for p in args.pool_types.split(",")]
        
    if not args.no_unknownCells:
        out_pat_dir = in_root / "patient_wise"
        fig_root    = in_root / "figures"
    else:
        out_pat_dir = in_root / "patient_wise_NoUnknown"
        fig_root    = in_root / "figures_NoUnknown"
    
    

    all_combined = []

    for pool_type in pool_types:
        print(f"\n{'='*70}")
        print(f"Pool type: {pool_type}")
        print(f"{'='*70}")

        # ── Load cell-level CSVs ───────────────────────────────────────────
        cell_data = load_cell_csvs(in_root, pool_type, args.no_unknownCells)
        if not cell_data:
            print(f"  [skip] No CSVs found for pool_type={pool_type}")
            continue
        print(f"  Loaded {len(cell_data)} (method, study) pairs")

        # ── Compute patient-wise metrics ───────────────────────────────────
        pat_dir = out_pat_dir / pool_type
        df_pat  = compute_all_patient_wise(cell_data, pat_dir, pool_type)
        if df_pat.empty:
            print(f"  [warn] No patient-wise data for pool_type={pool_type}")
            continue
        all_combined.append(df_pat)

        # ── Generate figures ───────────────────────────────────────────────
        if not args.no_plots:
            fig_dir = fig_root / pool_type
            fig_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  Generating figures → {fig_dir}")

            print("  [comparison heatmaps]")
            plot_comparison_heatmap(df_pat, fig_dir, pool_type)

            print("  [comparison strip+box]")
            plot_comparison_stripbox(df_pat, fig_dir, pool_type)

            print("  [ablation heatmap]")
            plot_ablation_heatmap(df_pat, fig_dir, pool_type)

            print("  [ablation barplot + per-patient scatter]")
            plot_ablation_barplot(df_pat, fig_dir, pool_type)

            print("  [ablation per-patient strip+box]")
            plot_ablation_per_patient_stripbox(df_pat, fig_dir, pool_type)

            #print("  [delta heatmap]")
            #plot_delta_heatmap(df_pat, fig_dir, pool_type)

            print("  [Fig 2+3: precision@k bar chart — moffitt]")
            #plot_topn_precision_bar(cell_data, fig_dir, pool_type)
            plot_study_precision_bar(cell_data, fig_dir, pool_type,
                             k_vals=(10, 20, 30, 40, 50),
                             ensemble_keys=STUDY_LEVEL_ENSEMBLE_KEYS,
                             ref_key_pref=("scTRP_simclr_zscoreMean_study", "scTRP_simclr"),
                             alternative="two-sided", study_="moffitt")
            plot_study_precision_bar(cell_data, fig_dir, pool_type,
                             k_vals=(10, 20, 30, 40, 50),
                             ensemble_keys=STUDY_LEVEL_ENSEMBLE_KEYS,
                             ref_key_pref=("scTRP_simclr_zscoreMean_study", "scTRP_simclr"),
                             alternative="two-sided", removeNoPositivePatient=True, study_="moffitt")
            plot_perpatient_precision_box(cell_data, fig_dir, pool_type,
                                  k_vals=(10, 20, 30, 40, 50),
                                  ensemble_keys=ALL_ENSEMBLE_KEYS,
                                  ref_key_pref=("scTRP_simclr_zscoreMean_study", "scTRP_simclr"),
                                  alternative="two-sided", study_="moffitt")
            plot_perpatient_precision_box(cell_data, fig_dir, pool_type,
                                  k_vals=(10, 20, 30, 40, 50),
                                  ensemble_keys=ALL_ENSEMBLE_KEYS,
                                  ref_key_pref=("scTRP_simclr_zscoreMean_study", "scTRP_simclr"),
                                  alternative="two-sided", removeNoPositivePatient=True, study_="moffitt")
            

    # ── Save combined CSV ──────────────────────────────────────────────────────
    if all_combined:
        df_all_combined = pd.concat(all_combined, ignore_index=True)
        out_path = in_root / "all_patient_wise.csv"
        if args.no_unknownCells:
            out_path = in_root / "all_patient_wise_NoUnknown.csv"
        df_all_combined.to_csv(out_path, index=False)
        print(f"\n{'='*70}")
        print(f"Combined CSV: {out_path}  ({len(df_all_combined)} rows)")
        print(f"Columns: pool_type, method, study, sample, + {len(METRIC_COLS)} metrics")

        # Print summary table
        summary = (df_all_combined[df_all_combined["method"].isin(SCTRP_KEYS + ["xgboost"])]
                   .groupby(["pool_type", "method", "study"])["ROC_AUC"]
                   .agg(["mean", "count"]).round(3))
        print("\nMean AUROC — scTRP methods by pool_type:")
        print(summary["mean"].unstack("study").to_string())
    else:
        print("\n[warn] No data collected — check in_root path and CSV files")

    print("\nDone.")


if __name__ == "__main__":
    main()
