#!/usr/bin/env python3
"""
collect_cell_predictions.py

Step 1 & 2 of unified prediction pipeline.

Collects raw cell-level prediction scores from all methods/studies and
applies 4 clonotype summarisation transforms:
  raw      — original per-cell scores (no pooling)
  mean     — mean score across cells sharing sample_clone_id
  max      — max score
  median   — median score
  p75      — 75th-percentile score

Methods covered
  benchmark:  AUCell_Lowery, Seurat_Caushi, Hanada, UCell_Meng, UCell_Oliveira, Zheng
  scTRP_simclr  — transcription + TCR clonotype (CK_SIMCLR per-study checkpoints)
  scTRP_only    — transcription only (single tvt6 checkpoint, all 6 studies)
  xgboost       — XGBoost (Chiffelle only)

Output layout
  {NEW_ROOT}/raw_cell_scores/{method_key}_{study}.csv
  {NEW_ROOT}/pooled_scores/{pool_type}/{method_key}_{study}.csv

Each CSV has columns:
  cell_barcode (index), score, reactivity, sample_clone_id, sample

Usage
-----
python collect_cell_predictions.py [--out_root DIR] [--by_flag cell|clonotype] [--score_rank]
"""

import argparse
import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = "/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein"
RES  = f"{BASE}/results/classifier"
JOBS = f"{BASE}/jobs/classifier/six_studies"

DEFAULT_NEW_ROOT = (
    f"{RES}/case_study/ICB_chiffelle_5case_inference_6AllasTrain/"
    "unified_cell_predictions"
)

# ── Log1p h5ads (for sample column mapping) ───────────────────────────────────
LOG1P_H5AD = {
    "caushi":    f"{BASE}/data_collection/caushi_nature_neoantigen_specificTIL/rawCount_adatas_cd8/caushi_4patients_log1p.h5ad",
    "hanada":    f"{BASE}/data_collection/hanada_NSCLC_CancerCell/runs/rawCount_adatas_cd8/hanada_4samples_CD8_log1p.h5ad",
    "lowery":    f"{BASE}/data_collection/lowery_neoTCR_metastatic/runs/rawCount_adatas_cd8/lowery_14samples_log1p_cd0718_agree_CD8.h5ad",
    "meng":      f"{BASE}/data_collection/meng_scienceTranslationalMedicine_PDAC/rawCount_adatas_cd8/GSE254250_PDAC_9sample_CD8_RNA_expression_meta_basicQC_log1p.h5ad",
    "oliveira":  f"{BASE}/data_collection/oliveira_melanoma_Nature/rawCount_adatas_cd8/Oliveira_log1p.h5ad",
    "zheng":     f"{BASE}/data_collection/zheng_GI_cancer_CancerCell/93cvs2z3mz-2/rawCount_adatas_cd8/zheng_5samples_log1p.h5ad",
    "chiffelle": f"{BASE}/data_collection/chiffelle_melanoma_Immunity/rawCount_adatas_cd8/chiffelle_15samples_CD8_log1p.h5ad",
    "moffitt":   f"{BASE}/data_collection/moffitt_NSCLC_Immunity/rawCount_adatas_cd8/moffitt_GSE255549_CD8_log1p.h5ad",
}

# ── Log1p .mtx paths (for benchmark meta CSV derivation) ─────────────────────
LOG1P_MTX = {
    "caushi":    f"{BASE}/data_collection/caushi_nature_neoantigen_specificTIL/rawCount_adatas_cd8/caushi_4patients_log1p.mtx",
    "hanada":    f"{BASE}/data_collection/hanada_NSCLC_CancerCell/runs/rawCount_adatas_cd8/hanada_4samples_CD8_log1p.mtx",
    "lowery":    f"{BASE}/data_collection/lowery_neoTCR_metastatic/runs/rawCount_adatas_cd8/lowery_14samples_log1p_cd0718_agree_CD8.mtx",
    "meng":      f"{BASE}/data_collection/meng_scienceTranslationalMedicine_PDAC/rawCount_adatas_cd8/GSE254250_PDAC_9sample_CD8_RNA_expression_meta_basicQC_log1p.mtx",
    "oliveira":  f"{BASE}/data_collection/oliveira_melanoma_Nature/rawCount_adatas_cd8/Oliveira_log1p.mtx",
    "zheng":     f"{BASE}/data_collection/zheng_GI_cancer_CancerCell/93cvs2z3mz-2/rawCount_adatas_cd8/zheng_5samples_log1p.mtx",
    "chiffelle": f"{BASE}/data_collection/chiffelle_melanoma_Immunity/rawCount_adatas_cd8/chiffelle_15samples_CD8_log1p.mtx",
    "moffitt":   f"{BASE}/data_collection/moffitt_NSCLC_Immunity/rawCount_adatas_cd8/moffitt_GSE255549_CD8_log1p.mtx",
}


ENSEMBLE_STUDIES = ["chiffelle", "moffitt"]

# ── Benchmark method configs ──────────────────────────────────────────────────
BENCHMARK_METHODS = {
    "AUCell_Lowery": {
        "score_col": "Lowery",
        "sep": ",",
        "get_score_path": lambda p: os.path.join(
            f"{RES}/benchmark_com_20240729_sixStudies_Lowery_signature" if not p in ENSEMBLE_STUDIES
            else f"{RES}/benchmark_com_20250225_sixStudies_Lowery_signature",
            f"{p}_AUC_Lowery_Meng.csv",
        ),
    },
    "Seurat_Caushi": {
        "score_col": "Caushi_Seurat1",
        "sep": "\t",
        "get_score_path": lambda p: os.path.join(
            f"{RES}/benchmark_com_20240809_sixStudies_Caushi_signature" if not p in ENSEMBLE_STUDIES
            else f"{RES}/benchmark_com_20250225_sixStudies_Caushi_signature",
            f"{p}_Seurat_Caushi_scores_{'0809' if not p in ENSEMBLE_STUDIES else '0225'}.csv",
        ),
    },
    "Hanada": {
        "score_col": "Hanada",
        "sep": ",",
        "get_score_path": lambda p: os.path.join(
            f"{RES}/benchmark_com_20240808_sixStudies_Hanada_signature" if not p in ENSEMBLE_STUDIES
            else f"{RES}/benchmark_com_20250225_sixStudies_Hanada_signature",
            f"{p}_Hanada_signature.csv",
        ),
    },
    "UCell_Meng": {
        "score_col": "TR30_UCell",
        "sep": "\t",
        "get_score_path": lambda p: os.path.join(
            f"{RES}/benchmark_com_20240809_sixStudies_Meng_signature" if not p in ENSEMBLE_STUDIES
            else f"{RES}/benchmark_com_20250225_sixStudies_Meng_signature",
            f"{p}_UCell_Meng_scores_{'0809' if not p in ENSEMBLE_STUDIES else '0225'}.csv",
        ),
    },
    "UCell_Oliveira": {
        "score_col": "Oliveira_Seurat1",
        "sep": "\t",
        "get_score_path": lambda p: os.path.join(
            f"{RES}/benchmark_com_20240809_sixStudies_Oliveira_signature" if not p in ENSEMBLE_STUDIES
            else f"{RES}/benchmark_com_20250225_sixStudies_Oliveira_signature",
            f"{p}_UCell_Oliveira_scores_{'0809' if not p in ENSEMBLE_STUDIES else '0225'}.csv",
        ),
    },
    "Zheng": {
        "score_col": "Zheng",
        "sep": ",",
        "get_score_path": lambda p: os.path.join(
            f"{RES}/benchmark_com_20240809_sixStudies_Zheng_signature" if not p in ENSEMBLE_STUDIES
            else f"{RES}/benchmark_com_20250225_sixStudies_Zheng_signature",
            f"{p}_Zheng_signature.csv",
        ),
    },
}

# ── scTRP checkpoint paths ────────────────────────────────────────────────────
# scTRP_simclr: one dedicated leave-one-out checkpoint per study (+ TCR clonotype)
CK_SIMCLR = {
    "caushi":   f"{JOBS}/supconloss_simCLR/test_caushi/hvg2000_by_study_6_89_genes/regenerate_pt/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght1.0-SuContinue/model_e3.pt",
    "hanada":   f"{JOBS}/supconloss_simCLR/test_hanada/hvg2000_by_study_6_89_genes/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght0.3-SuContinue/model_e25.pt",
    "lowery":   f"{JOBS}/supconloss_simCLR/test_lowery/hvg2000_by_study_6_89_genes/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght1.0-SuContinue/model_e1.pt",
    "meng":     f"{JOBS}/supconloss_simCLR/test_meng/hvg2000_by_study_6_89_genes_newCK_lessLayer/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght0.3-SuContinue/model_e10.pt",
    "oliveira": f"{JOBS}/supconloss_simCLR/test_oliveira/hvg2000_by_study_6_89_genes_newCK_lessLayer/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght1.0-SuContinue/model_e5.pt",
    "zheng":    f"{JOBS}/supconloss_simCLR/test_zheng/hvg2000_by_study_6_89_genes/regenerate_pt/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght0.3-SuContinue/model_e1.pt",
}

# scTRP_only: single tvt6 checkpoint — transcription only (no TCR clonotype)
CK_ONLY = {
    "caushi": f'{JOBS}/supconloss_union_intersection/test_caushi/hvg2000_by_study_6_89_genes/save/dev_SixStudy_supconH_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_npos9_nneg120_t0.1_maxl100-SuContinue/model_e39.pt',
    'hanada': f'{JOBS}/supconloss_union_intersection/test_hanada/hvg2000_by_study_6_89_genes/save/dev_SixStudy_supconH_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_npos9_nneg120_t0.1_maxl100-SuContinue/model_e30.pt',
    'lowery': f'{JOBS}/supconloss_union_intersection/test_lowery/hvg2000_by_study_6_89_genes/save/dev_SixStudy_supconH_glenhvg2000_by_study_6_89_genes_unfix2_lr0.0001_npos9_nneg120_t0.1_maxl100-SuContinue/model_e20.pt',
    'meng': f'{JOBS}/supconloss_union_intersection/test_meng/hvg2000_by_study_6_89_genes/save/dev_SixStudy_supconH_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_npos9_nneg120_t0.1_maxl100-SuContinue/latest_ck.pt',
    'oliveira': f'{JOBS}/supconloss_union_intersection/test_oliveira/hvg2000_by_study_6_89_genes/save/dev_SixStudy_supconH_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_npos9_nneg120_t0.1_maxl100-SuContinue/model_e40.pt',
    'zheng': f'{JOBS}/supconloss_union_intersection/test_zheng/hvg2000_by_study_6_89_genes/save/dev_SixStudy_supconH_glenhvg2000_by_study_6_89_genes_unfix12_lr0.0001_npos9_nneg120_t0.1_maxl100-SuContinue/model_e2.pt',
 }

SCORE_COL_H5AD = "pro_jenks_OT_deltarho_30best_score"
POOL_TYPES = ["raw", "mean", "max", "median", "p75"]
ENSEMBLE_MODES = ["score_mean", "rank_sum", "vote_fraction", "rank_majority"]
SIX_STUDIES = ["caushi", "hanada", "lowery", "meng", "oliveira", "zheng"]

ALL_STUDIES = SIX_STUDIES + ENSEMBLE_STUDIES

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_sample_col(obs_df):
    for cname in ["sample", "patient", "batch", "donor", "Sample"]:
        if cname in obs_df.columns:
            return obs_df[cname]
    if "sample_clone_id" in obs_df.columns:
        v0 = obs_df["sample_clone_id"].iloc[0]
        if "_TRA_" in v0:
            return obs_df["sample_clone_id"].str.split("_TRA_").str[0]
        return obs_df["sample_clone_id"].str.rsplit("_", n=2).str[0]
    raise ValueError(f"Cannot find sample column. Available: {list(obs_df.columns)}")


def find_prj_avg_h5ad(ck_path, glob_pattern):
    ck_dir  = os.path.dirname(ck_path)
    matches = glob.glob(os.path.join(ck_dir, glob_pattern))
    if not matches:
        return None
    matches.sort(key=os.path.getctime)
    return matches[-1]


def find_raw_h5ad(ck_path, study_prefix):
    ck_dir  = os.path.dirname(ck_path)
    matches = [f for f in glob.glob(os.path.join(ck_dir, f"*{study_prefix}*.h5ad"))
               if ("_averagedPRJ" not in f) and ("_attn" not in f) and ("val" not in f) and ("train" not in f) and ("caushi_other" not in f)]
    if not matches:
        return None
    matches.sort(key=os.path.getctime)
    return matches[-1]


def epoch_from_ck(ck_path):
    m = re.search(r"model_e(\d+)\.", ck_path)
    return m.group(1) if m else "?"


def apply_pooling(df, pool_type):
    """Apply clonotype-level score pooling via groupby transform (keeps cell-level index)."""
    if pool_type == "raw" or "sample_clone_id" not in df.columns:
        return df.copy()
    df = df.copy()
    grp = df.groupby("sample_clone_id")["score"]
    if pool_type == "mean":
        df["score"] = grp.transform("mean")
    elif pool_type == "max":
        df["score"] = grp.transform("max")
    elif pool_type == "median":
        df["score"] = grp.transform("median")
    elif pool_type == "p75":
        df["score"] = grp.transform(lambda x: x.quantile(0.75))
    return df


def save_all_pools(df, raw_dir, pool_dirs, key, study, pool_types=POOL_TYPES):
    """Save raw CSV and all pooled CSVs for one (method_key, study) pair."""
    raw_path = raw_dir / f"{key}_{study}.csv"
    df.to_csv(raw_path)

    for pt in pool_types:
        if pt == "raw":
            continue
        df_pool = apply_pooling(df, pt)
        out = pool_dirs[pt] / f"{key}_{study}.csv"
        df_pool.to_csv(out)

    n = len(df)
    n_clones = df["sample_clone_id"].nunique() if "sample_clone_id" in df.columns else "?"
    print(f"    saved: {n} cells | {n_clones} clonotypes → {raw_path.name}")


# ── Extractor: benchmark CSV + meta CSV ───────────────────────────────────────

def extract_benchmark(score_path, meta_path, score_col, sep=",", exclude_sample_list=[]):
    if not os.path.exists(score_path):
        print(f"    [skip] score file missing: {score_path}")
        return None
    if not os.path.exists(meta_path):
        print(f"    [skip] meta file missing: {meta_path}")
        return None
    meta_df  = pd.read_csv(meta_path, index_col=0)
    score_df = pd.read_csv(score_path, header=0, index_col=0, sep=sep)

    if score_col not in score_df.columns:
        available = list(score_df.columns)
        print(f"    [skip] score_col {score_col!r} not found; available: {available}")
        return None
    score_df = score_df[[score_col]].rename(columns={score_col: "score"})

    try:
        sample_s = get_sample_col(meta_df)
    except ValueError as e:
        print(f"    [skip] {e}")
        return None

    df = meta_df[["reactivity", "sample_clone_id", "reactivity_fine"]].copy()
    df["sample"] = sample_s.values
    df["score"]  = score_df.reindex(df.index)["score"].values
    df.index.name = "cell_barcode"
    df = df.dropna(subset=["score", "reactivity"])
    df["reactivity"] = df["reactivity"].astype(int)
    df["reactivity_fine"] = df["reactivity_fine"].astype(int)
    df = df[~df['sample'].isin(exclude_sample_list)]
    return df[["score", "reactivity", "sample_clone_id", "sample", "reactivity_fine"]]


# ── Extractor: scTRP h5ad (single checkpoint, one study) ─────────────────────
#check if "score" needs to be groupby sample first. Conclusion: default to save the original scTRP prediction score. Can group later. No need to rank. 
def extract_sctrp_single(ck_path, study, by_flag, ori_obs, score_rank=False):
    epoch = epoch_from_ck(ck_path)
    if by_flag == "clonotype":
        h5ad_path = find_prj_avg_h5ad(ck_path, f"*{study}*_e{epoch}*_averagedPRJ.h5ad")
    else:
        h5ad_path = find_raw_h5ad(ck_path, study)

    if h5ad_path is None:
        print(f"    [skip] {study}: h5ad not found in {os.path.dirname(ck_path)}")
        return None

    import scanpy as sc
    adata = sc.read_h5ad(h5ad_path)
    if SCORE_COL_H5AD not in adata.obs.columns:
        print(f"    [skip] {study}: {SCORE_COL_H5AD!r} absent. "
              f"obs cols: {list(adata.obs.columns)[:8]}")
        return None


    # Since I have updated the 'sample_clone_id', need to read from the ori_obs
    #if "sample_clone_id" not in adata.obs.columns:
    adata.obs["sample_clone_id"] = ori_obs["sample_clone_id"].reindex(adata.obs_names)
    adata.obs["reactivity_fine"] = ori_obs["reactivity_fine"].reindex(adata.obs_names)
    print(adata.obs['sample_clone_id'].notna().sum(), '/', len(adata), 'cells have a clone ID')

    df = adata.obs[[SCORE_COL_H5AD, "reactivity", "sample_clone_id", "reactivity_fine"]].copy()
    df.rename(columns={SCORE_COL_H5AD: "score"}, inplace=True)
    if score_rank:
        df["score"] = df["score"].rank(method="average", ascending=True)
    df["reactivity"] = (df["reactivity"].astype(str).str.strip()
                        .map({"1": 1, "0": 0, 1: 1, 0: 0}))
    try:
        sample_s = get_sample_col(ori_obs)
    except ValueError:
        sample_s = get_sample_col(adata.obs)
    df["sample"] = sample_s.reindex(df.index).values
    df.index.name = "cell_barcode"
    df = df.dropna(subset=["score", "reactivity", "sample"])
    df["reactivity"] = df["reactivity"].astype(int)
    df["reactivity_fine"] = df["reactivity_fine"].astype(int)
    return df[["score", "reactivity", "sample_clone_id", "sample", "reactivity_fine"]]


# ── Chiffelle ensemble helpers ────────────────────────────────────────────────

def _jenks_thresh_simple(vals):
    """Per-patient Jenks threshold; falls back to median if jenkspy unavailable."""
    try:
        import jenkspy
        return jenkspy.jenks_breaks(list(vals), n_classes=2)[1]
    except Exception:
        return float(np.median(vals))


def _pool_score_matrix(score_mat, sample_clone_ids, pool_type):
    """Pool each checkpoint column by clonotype for one pool_type."""
    if pool_type == "raw" or sample_clone_ids is None or sample_clone_ids.isna().all():
        return score_mat.copy()
    pooled = {}
    clones = sample_clone_ids.reindex(score_mat.index)
    for col in score_mat.columns:
        tmp = pd.Series(score_mat[col].values, index=clones.values)
        grp = tmp.groupby(level=0)
        if pool_type == "mean":
            p = grp.transform("mean")
        elif pool_type == "max":
            p = grp.transform("max")
        elif pool_type == "median":
            p = grp.transform("median")
        elif pool_type == "p75":
            p = grp.transform(lambda x: x.quantile(0.75))
        else:
            p = tmp
        pooled[col] = p.values
    return pd.DataFrame(pooled, index=score_mat.index)







def plot_split_violin(
    df, score_col="score", patient_col="sample", reactivity_col="reactivity",
    title=None, order=None, figsize=None, save_prefix=None,
):
    """
    Split violin on a SHARED score axis: for each patient, left half = reactivity==0,
    right half = reactivity==1. Only valid for cross-patient-comparable score modes.
    """
    # reactivity 0/1 → your palette (0 = non-reactive/viral, 1 = antitumor-reactive)
    SPLIT_PALETTE = {0: "#5688C7", 1: "#E8904A"}

    d = df[[score_col, patient_col, reactivity_col]].dropna().copy()
    d[reactivity_col] = d[reactivity_col].astype(int)

    patients = order if order is not None else sorted(d[patient_col].unique())
    d[patient_col] = pd.Categorical(d[patient_col], categories=patients, ordered=True)
    n_pat = len(patients)

    if figsize is None:
        figsize = (max(6, 0.7 * n_pat + 2), 5)
    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=d, x=patient_col, y=score_col, hue=reactivity_col,
        split=True, inner="quartile", density_norm="count", cut=0,
        palette=SPLIT_PALETTE, hue_order=[0, 1], order=patients,
        linewidth=0.8, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel(score_col)
    if title:
        ax.set_title(title)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")

    # relabel legend 0/1 → readable
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["reactivity == 0", "reactivity == 1"],
              title=None, frameon=False, loc="upper right", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()

    if save_prefix:
        for ext in ("pdf", "png", "svg"):
            fig.savefig(f"{save_prefix}.{ext}", dpi=300, bbox_inches="tight")
    return fig, ax


def _jenks_vote(mat, group_s=None):
    """
    Per-checkpoint Jenks(k=2) binary vote → fraction across checkpoints → majority.

    group_s is None  → one threshold per checkpoint over ALL cells   (study-wide)
    group_s provided → one threshold per checkpoint PER GROUP         (per-patient)

    Returns (vote_fraction, vote_pred). NaN cells vote 0; checkpoints/groups with
    <2 valid values contribute 0 (same convention as the original vote block).
    """
    vm = pd.DataFrame(0.0, index=mat.index, columns=mat.columns)
    for col in mat.columns:
        s = mat[col]
        if group_s is None:
            vals = s.dropna().values
            if len(vals) < 2:
                continue
            thresh = _jenks_thresh_simple(vals)
            vm[col] = (s > thresh).astype(float)          # NaN > thresh → False → 0
        else:
            for pat, grp in s.groupby(group_s):
                vals = grp.dropna().values
                if len(vals) < 2:
                    continue
                thresh = _jenks_thresh_simple(vals)
                vm.loc[grp.index, col] = (grp.values > thresh).astype(float)
    frac = vm.mean(axis=1)
    return frac, (frac > 0.5).astype(int)
    

"""
def _compute_chiffelle_ensembles(score_mat_pooled, sample_s):
    "
    Given a pooled score matrix (n_cells × n_checkpoints) and patient labels,
    return 4 ensemble modes as (score_series, pred_series_or_None) pairs.

    Modes
    -----
    score_mean    : mean of pooled scores → Jenks downstream
    rank_sum      : per-patient Borda rank sum → Jenks downstream
    vote_fraction : per-ck per-patient Jenks binary → mean as score, majority (>0.5) as pred
    rank_majority : rank_sum as score + majority vote as pred
    "
    # 1. score_mean
    score_mean = score_mat_pooled.mean(axis=1)

    # 2. rank_sum — per-patient Borda ranking (scale-invariant)
    rank_sum = pd.Series(0.0, index=score_mat_pooled.index)
    for col in score_mat_pooled.columns:
        pat_ranks = score_mat_pooled[col].groupby(sample_s).rank(
            method="average", ascending=True
        )
        rank_sum += pat_ranks.fillna(0)

    # 3+4. per-checkpoint per-patient Jenks binary → majority vote
    vote_mat = pd.DataFrame(
        0.0, index=score_mat_pooled.index, columns=score_mat_pooled.columns
    )
    for col in score_mat_pooled.columns:
        for pat, grp in score_mat_pooled[col].groupby(sample_s):
            vals = grp.dropna().values
            if len(vals) < 2:
                continue
            thresh = _jenks_thresh_simple(vals)
            vote_mat.loc[grp.index, col] = (grp.values > thresh).astype(float)
    vote_fraction = vote_mat.mean(axis=1)
    vote_pred     = (vote_fraction > 0.5).astype(int)

    return {
        "score_mean":    (score_mean,    None),       # pred via Jenks downstream
        "rank_sum":      (rank_sum,      None),       # pred via Jenks downstream
        "vote_fraction": (vote_fraction, vote_pred),  # pred = majority vote
        "rank_majority": (rank_sum,      vote_pred),  # score=rank_sum, pred=majority vote
    }
"""



def _compute_chiffelle_ensembles(score_mat_pooled, sample_s):
    """
    Given a pooled score matrix (n_cells × n_checkpoints) and patient labels,
    return ensemble modes as (score_series, pred_series_or_None) pairs.

    Modes
    -----
    score_mean          : mean of pooled scores → Jenks downstream
    scoreMean_majority  : mean of pooled scores as score + majority vote as pred
    rank_sum            : per-patient Borda rank sum → Jenks downstream
    rankSum_study       : study-wide Borda rank sum → Jenks downstream
    rankProduct_patient : per-patient rank geometric mean → Jenks downstream
    rankProduct_study   : study-wide rank geometric mean → Jenks downstream
    rankMin_patient     : per-patient consensus (worst) rank → Jenks downstream
    rankMin_study       : study-wide consensus (worst) rank → Jenks downstream
    zscoreMean_patient  : per-ck per-patient z-scored, then mean → Jenks downstream
    zscoreMean_study    : per-ck study-wide z-scored, then mean → Jenks downstream
    vote_fraction       : per-ck per-patient Jenks binary → mean as score, majority (>0.5) as pred
    rank_majority       : rank_sum as score + majority vote as pred

    All score/rank modes are oriented so that HIGHER = more reactive.
    """
    n_ck = score_mat_pooled.shape[1]

    # 1. score_mean
    score_mean = score_mat_pooled.mean(axis=1)

    # 2. Rank-based modes — build per-patient and study-wide rank matrices once.
    #    ascending=True: low score → low rank → higher aggregated rank = more reactive.
    rank_mat_patient = pd.DataFrame(
        index=score_mat_pooled.index, columns=score_mat_pooled.columns, dtype=float
    )
    rank_mat_study = pd.DataFrame(
        index=score_mat_pooled.index, columns=score_mat_pooled.columns, dtype=float
    )
    for col in score_mat_pooled.columns:
        rank_mat_patient[col] = score_mat_pooled[col].groupby(sample_s).rank(
            method="average", ascending=True
        )
        rank_mat_study[col] = score_mat_pooled[col].rank(
            method="average", ascending=True
        )

    # 2a. rank_sum family — sum of ranks (missing checkpoint → 0, i.e. pushed low)
    rank_sum       = rank_mat_patient.fillna(0.0).sum(axis=1)
    rank_sum_study = rank_mat_study.fillna(0.0).sum(axis=1)

    # 2b. rank_product family — geometric mean of ranks, computed in log space
    rank_product_patient = np.exp(
        np.log(rank_mat_patient.fillna(1.0)).sum(axis=1) / n_ck
    )
    rank_product_study = np.exp(
        np.log(rank_mat_study.fillna(1.0)).sum(axis=1) / n_ck
    )

    # 2c. rank_min family — consensus: worst (lowest) rank across checkpoints
    rank_min_patient = rank_mat_patient.min(axis=1)
    rank_min_study   = rank_mat_study.min(axis=1)

    # 3. z-score-mean modes — standardize each checkpoint (ddof=0), then average
    zmat_patient = pd.DataFrame(
        index=score_mat_pooled.index, columns=score_mat_pooled.columns, dtype=float
    )
    zmat_study = pd.DataFrame(
        index=score_mat_pooled.index, columns=score_mat_pooled.columns, dtype=float
    )
    for col in score_mat_pooled.columns:
        zmat_patient[col] = score_mat_pooled[col].groupby(sample_s).transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
        s = score_mat_pooled[col]
        zmat_study[col] = (s - s.mean()) / s.std(ddof=0)
    zscore_mean_patient = zmat_patient.mean(axis=1)
    zscore_mean_study   = zmat_study.mean(axis=1)

    # 4. per-checkpoint per-patient Jenks binary → majority vote
    vote_mat = pd.DataFrame(
        0.0, index=score_mat_pooled.index, columns=score_mat_pooled.columns
    )
    for col in score_mat_pooled.columns:
        for pat, grp in score_mat_pooled[col].groupby(sample_s):
            vals = grp.dropna().values
            if len(vals) < 2:
                continue
            thresh = _jenks_thresh_simple(vals)
            vote_mat.loc[grp.index, col] = (grp.values > thresh).astype(float)
    vote_fraction = vote_mat.mean(axis=1)
    vote_pred     = (vote_fraction > 0.5).astype(int)

    # z-score-based votes: study-wide threshold (matches the cross-patient
    # normalization; a per-patient threshold here would reduce to `vote_pred`).
    _, z_vote_pred_patient = _jenks_vote(zmat_patient, group_s=sample_s)
    _, z_vote_pred_study   = _jenks_vote(zmat_study,   group_s=None)
    _, z_vote_pred_studyP   = _jenks_vote(zmat_study,   group_s=sample_s)

    return {
        "score_mean":          (score_mean,           None),        # pred via Jenks downstream
        "scoreMean_majority":  (score_mean,           vote_pred),   # score=mean, pred=majority vote
        "rank_sum":            (rank_sum,             None),        # pred via Jenks downstream
        "rankSum_study":       (rank_sum_study,       None),        # pred via Jenks downstream
        "rankProduct_patient": (rank_product_patient, None),        # pred via Jenks downstream
        "rankProduct_study":   (rank_product_study,   None),        # pred via Jenks downstream
        "rankMin_patient":     (rank_min_patient,     None),        # pred via Jenks downstream
        "rankMin_study":       (rank_min_study,       None),        # pred via Jenks downstream
        "zscoreMean_patient":  (zscore_mean_patient,  z_vote_pred_patient),        
        "zscoreMean_study":    (zscore_mean_study,    z_vote_pred_study),    # score=zscore_mean_study, pred=majority vote with uni threshold per study  
        "zscoreMean_studyPerPatient":    (zscore_mean_study,    z_vote_pred_studyP),    # score=zscore_mean_study, pred=majority vote with threshold per patient 
        "vote_fraction":       (vote_fraction,        vote_pred),   # pred = majority vote
        "rank_majority":       (rank_sum,             vote_pred),   # score=rank_sum, pred=majority vote
    }




def extract_and_save_chiffelle_ensembles(
        ck_dic, study_list, ori_obs, method_prefix, raw_dir, pool_dirs, study_to_test, exclude_sample_list):
    """
    Load 6 checkpoint raw h5ads for study_to_test once; apply each pool_type to each
    checkpoint's scores; compute 4 ensemble modes; save CSVs.

    Output keys: {method_prefix}_{mode}_{study_to_test}
    """
    import scanpy as sc

    score_frames = []
    ref_obs = None

    SAME_AXIS_MODES = [
        "score_mean", "scoreMean_majority", "rankSum_study",
        "rankProduct_study", "rankMin_study", "zscoreMean_study",
    ]

    for study in study_list:
        ck_path = ck_dic.get(study)
        if not ck_path or not os.path.exists(ck_path):
            print(f"    [warn] {method_prefix}/{study}: checkpoint missing")
            continue
        # Always use raw (cell-level) h5ad — pooling is explicit below
        h5ad_path = find_raw_h5ad(ck_path, study_to_test)
        if h5ad_path is None:
            print(f"    [warn] {method_prefix}/{study}: {study_to_test} h5ad not found in "
                  f"{os.path.dirname(ck_path)}")
            continue
        adata = sc.read_h5ad(h5ad_path)
        if SCORE_COL_H5AD not in adata.obs.columns:
            print(f"    [warn] {method_prefix}/{study}: {SCORE_COL_H5AD!r} absent")
            continue
        # Since I have updated the 'sample_clone_id', need to read from the ori_obs
        #if "sample_clone_id" not in adata.obs.columns:
        adata.obs["sample_clone_id"] = ori_obs["sample_clone_id"].reindex(adata.obs_names)
        adata = adata[~adata.obs['sample'].isin(exclude_sample_list)].copy()
        s = adata.obs[SCORE_COL_H5AD].rename(study)
        score_frames.append(s)
        if ref_obs is None:
            ref_obs = adata.obs

    if not score_frames or ref_obs is None:
        print(f"    [skip] {method_prefix}/{study_to_test}: no h5ads loaded")
        return

    # Cell-level score matrix (n_cells × n_checkpoints)
    score_mat = pd.concat(score_frames, axis=1)
    print(f"    {method_prefix}/{study_to_test}: {len(score_frames)} checkpoints, "
          f"{len(score_mat)} cells")

    # Meta
    sample_clone_ids = ori_obs["sample_clone_id"].reindex(score_mat.index)
    try:
        sample_s = get_sample_col(ori_obs).reindex(score_mat.index)
    except ValueError:
        sample_s = get_sample_col(ref_obs).reindex(score_mat.index)
    reactivity = ori_obs["reactivity"].reindex(score_mat.index)
    reactivity = (reactivity.astype(str).str.strip()
                  .map({"1": 1, "0": 0, 1: 1, 0: 0}))
    reactivity_fine = ori_obs["reactivity_fine"].reindex(score_mat.index)

    for pool_type in POOL_TYPES:
        score_mat_pooled  = _pool_score_matrix(score_mat, sample_clone_ids, pool_type)
        ensemble_results  = _compute_chiffelle_ensembles(score_mat_pooled, sample_s)

        for mode, (score_s, pred_s) in ensemble_results.items():
            df = pd.DataFrame({
                "score":          score_s,
                "reactivity":     reactivity,
                "sample_clone_id": sample_clone_ids,
                "sample":         sample_s,
                "reactivity_fine": reactivity_fine,
                
            }, index=score_mat.index)
            if pred_s is not None:
                df["pred"] = pred_s.values
            df.index.name = "cell_barcode"
            df = df.dropna(subset=["score", "reactivity", "sample"])
            df["reactivity"] = df["reactivity"].astype(int)
            df["reactivity_fine"] = df["reactivity_fine"].astype(int)
            cols = ["score", "reactivity", "sample_clone_id", "sample", "reactivity_fine"]
            if pred_s is not None:
                cols.append("pred")
            df = df[cols]

            key = f"{method_prefix}_{mode}"
            if pool_type == "raw":
                out_dir = raw_dir
            else:
                out_dir = pool_dirs[pool_type]
            out = out_dir / f"{key}_{study_to_test}.csv"
            df.to_csv(out)
            n_cl = df["sample_clone_id"].nunique() if "sample_clone_id" in df.columns else "?"
            print(f"      [{pool_type}][{mode}] {len(df)} cells | {n_cl} clones → {out.name}")
            if mode in SAME_AXIS_MODES:
                print(f"      [{pool_type}][{mode}]  → f{str(out_dir / f'violin_{method_prefix}_{mode}')}")
                plot_split_violin(df, title=f"{method_prefix} · {mode}", save_prefix=str(out_dir / f"violin_{method_prefix}_{mode}"))


# ── Extractor: XGBoost prediction CSV ────────────────────────────────────────

def extract_xgboost(csv_path, h5ad_path, exclude_sample_list):
    if not os.path.exists(csv_path):
        print(f"    [skip] XGBoost CSV missing: {csv_path}")
        return None
    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "cell_barcode"

    if "xgboost_score" not in df.columns:
        print(f"    [skip] 'xgboost_score' not found. cols: {list(df.columns)}")
        return None

    df = df[["xgboost_score", "reactivity", "sample"]].rename(
        columns={"xgboost_score": "score"}
    ).copy()
    df["reactivity"] = pd.to_numeric(df["reactivity"], errors="coerce").astype("Int64")

    # Add sample_clone_id from h5ad
    if h5ad_path and os.path.exists(h5ad_path):
        import scanpy as sc
        adata = sc.read_h5ad(h5ad_path, backed="r")
        if "sample_clone_id" in adata.obs.columns:
            df["sample_clone_id"] = adata.obs["sample_clone_id"].reindex(df.index).values
        else:
            df["sample_clone_id"] = df.index  # fallback
        if "reactivity_fine" in adata.obs.columns:
            df["reactivity_fine"] = adata.obs["reactivity_fine"].reindex(df.index).values
    else:
        df["sample_clone_id"] = df.index

    df = df.dropna(subset=["score", "reactivity"])
    df["reactivity"] = df["reactivity"].astype(int)
    df["reactivity_fine"] = df["reactivity_fine"].astype(int)
    df = df[~df['sample'].isin(exclude_sample_list)]
    return df[["score", "reactivity", "sample_clone_id", "sample", "reactivity_fine"]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect cell-level predictions (all methods/studies)")
    parser.add_argument("--out_root",  default=DEFAULT_NEW_ROOT,
                        help="Root directory for output CSVs")
    parser.add_argument("--by_flag",   choices=["cell", "clonotype"], default="cell",
                        help="Feature level for scTRP h5ads: cell or clonotype")
    parser.add_argument("--xgb_csv",   default=None,
                        help="Path to xgboost per-cell prediction CSV (xgb_csv_path only)")
    parser.add_argument("--score_rank", action="store_true",
                        help="Replace raw score with global rank (ascending, average ties) "
                             "before saving; useful for scale-invariant comparisons")
    args = parser.parse_args()

    new_root = Path(args.out_root)
    raw_dir  = new_root / "raw_cell_scores"
    raw_dir.mkdir(parents=True, exist_ok=True)

    
    exclude_sample_list = ["moffitt_Pt3_Aph", "moffitt_Pt3_PD_Tumor", "moffitt_Pt3_TD",
                           "moffitt_Pt31_Baseline_Tumor", "moffitt_Pt9_Baseline_Tumor", "moffitt_Pt3_Baseline_Tumor"]
    exclude_sample_list = ["moffitt_Pt3_Aph", "moffitt_Pt3_PD_Tumor"]

    exclude_sample_list = ["moffitt_Pt3_Aph", "moffitt_Pt3_PD_Tumor", "moffitt_Pt3_TD",
                           "moffitt_Pt3_TIL", "moffitt_Pt25_TIL", "moffitt_Pt14_TIL", "moffitt_Pt31_TIL", 
                           "moffitt_Pt9_TIL", "moffitt_Pt5_TIL", "moffitt_Pt7_TIL", "moffitt_Pt16_TIL"]
    # TD + baseline tumor
    exclude_sample_list = ["moffitt_Pt3_Aph", "moffitt_Pt3_PD_Tumor", 
                           "moffitt_Pt3_TIL", "moffitt_Pt25_TIL", "moffitt_Pt14_TIL", "moffitt_Pt31_TIL",
                           "moffitt_Pt9_TIL", "moffitt_Pt5_TIL", "moffitt_Pt7_TIL", "moffitt_Pt16_TIL"]
    
    pool_dirs = {}
    for pt in POOL_TYPES:
        if pt == "raw":
            continue
        d = new_root / "pooled_scores" / pt
        d.mkdir(parents=True, exist_ok=True)
        pool_dirs[pt] = d

    xgb_csv_path = {
        'chiffelle': f"{RES}/XGBoost_lightModel_out/chiffelle_xgboost_per_cell_predictions_bycell.csv",
        'moffitt': f"{RES}/XGBoost_lightModel_out/moffitt_GSE255549_CD8_log1p_xgboost_per_cell_predictions_bycell.csv"
    }

    # ========================================================================= #
    # The following has been generated, skip for now
    # ========================================================================= #


    # ── 1. Benchmark methods (6 methods × 7 studies) ─────────────────────────
    print("\n" + "="*70)
    print("STEP 1: Benchmark methods")
    print("="*70)

    for method_name, cfg in BENCHMARK_METHODS.items():
        print(f"\n  {method_name}")
        for study in ALL_STUDIES:
            score_path = cfg["get_score_path"](study)
            meta_path  = LOG1P_MTX[study].replace(".mtx", "_meta.csv")
            df = extract_benchmark(score_path, meta_path, cfg["score_col"], cfg["sep"], exclude_sample_list)
            if df is None:
                continue
            key = f"benchmark_{method_name}"
            save_all_pools(df, raw_dir, pool_dirs, key, study)

    # ── 2. scTRP_simclr (six held-out studies + chiffelle ensemble) ──────────
    print("\n" + "="*70)
    print("STEP 2: scTRP_simclr (transcription + TCR clonotype)")
    print("="*70)

    for study in SIX_STUDIES:
        ck_path = CK_SIMCLR.get(study)
        if not ck_path or not os.path.exists(ck_path):
            print(f"  [skip] {study}: checkpoint missing")
            continue
        if not os.path.exists(LOG1P_H5AD[study]):
            print(f"  [skip] {study}: log1p h5ad missing")
            continue
        print(f"\n  scTRP_simclr / {study}")
        import scanpy as sc
        ori_obs = sc.read_h5ad(LOG1P_H5AD[study], backed="r").obs
        df = extract_sctrp_single(ck_path, study, args.by_flag, ori_obs,
                                   score_rank=args.score_rank)
        if df is None:
            continue
        save_all_pools(df, raw_dir, pool_dirs, "scTRP_simclr", study)

    for study_to_test in ENSEMBLE_STUDIES:
        # Chiffelle: 6-checkpoint ensemble → 4 modes × 5 pool_types
        print(f"\n  scTRP_simclr / {study_to_test} (6 checkpoints × 4 ensemble modes)")
        if os.path.exists(LOG1P_H5AD[f"{study_to_test}"]):
            import scanpy as sc
            ori_chif_obs = sc.read_h5ad(LOG1P_H5AD[f"{study_to_test}"], backed="r").obs
            
            extract_and_save_chiffelle_ensembles(
                CK_SIMCLR, SIX_STUDIES, ori_chif_obs, "scTRP_simclr", raw_dir, pool_dirs , study_to_test, exclude_sample_list
            )
        

    # ── 3. scTRP_only (tvt6 checkpoint, six held-out studies + chiffelle ensemble) ─
    print("\n" + "="*70)
    print("STEP 3: scTRP_only (transcription only)")
    print("="*70)

    for study in SIX_STUDIES:
        ck_path = CK_ONLY.get(study)
        if not ck_path or not os.path.exists(ck_path):
            print(f"  [skip] {study}: checkpoint missing")
            continue
        if not os.path.exists(LOG1P_H5AD[study]):
            print(f"  [skip] {study}: log1p h5ad missing")
            continue
        print(f"\n  scTRP_only / {study}")
        import scanpy as sc
        ori_obs = sc.read_h5ad(LOG1P_H5AD[study], backed="r").obs
        df = extract_sctrp_single(ck_path, study, args.by_flag, ori_obs,
                                   score_rank=args.score_rank)
        if df is None:
            continue
        save_all_pools(df, raw_dir, pool_dirs, "scTRP_only", study)

    for study_to_test in ENSEMBLE_STUDIES:
        # Chiffelle: scTRP_only — same 4 modes × 5 pool_types
        print(f"\n  scTRP_only / {study_to_test} (6 checkpoints × 4 ensemble modes)")
        if os.path.exists(LOG1P_H5AD[f"{study_to_test}"]):
            import scanpy as sc
            ori_chif_obs = sc.read_h5ad(LOG1P_H5AD[f"{study_to_test}"], backed="r").obs
            
            extract_and_save_chiffelle_ensembles(
                CK_ONLY, SIX_STUDIES, ori_chif_obs, "scTRP_only", raw_dir, pool_dirs, study_to_test, exclude_sample_list
            )

    # ========================================================================= #

    
    # ── 4. XGBoost (ENSEMBLE_STUDIES only) ───────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 4: XGBoost (ENSEMBLE_STUDIES only)")
    for study_to_test in ENSEMBLE_STUDIES:
        print("="*70)
        print(f"  xgb_csv: {xgb_csv_path[f'{study_to_test}']}")
        df = extract_xgboost(xgb_csv_path[f'{study_to_test}'], LOG1P_H5AD[f"{study_to_test}"], exclude_sample_list)
        if df is not None:
            save_all_pools(df, raw_dir, pool_dirs, "xgboost", f"{study_to_test}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("DONE. Output layout:")
    print(f"  {new_root}/")
    print(f"    raw_cell_scores/          — {len(list(raw_dir.glob('*.csv')))} CSVs")
    for pt in POOL_TYPES:
        if pt == "raw":
            continue
        d = pool_dirs[pt]
        print(f"    pooled_scores/{pt}/  — {len(list(d.glob('*.csv')))} CSVs")
    print(f"\n  Schema per CSV:")
    print(f"    cell_barcode (index) | score | reactivity | sample_clone_id | sample [| pred]")
    print(f"    (pred column present for vote_fraction and rank_majority modes)")
    print(f"\n  Chiffelle ensemble modes: {ENSEMBLE_MODES}")
    print(f"\n  Next step: run compute_patient_metrics_all.py --in_root {new_root}")


if __name__ == "__main__":
    main()
