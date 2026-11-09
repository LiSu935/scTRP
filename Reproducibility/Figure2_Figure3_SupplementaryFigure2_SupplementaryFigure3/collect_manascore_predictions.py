#!/usr/bin/env python3
"""

# this is for manascore results, based on the 'collect_trace_predictions.py'. Note that manascore provide two scores. I will use 'score_i' for now, but it is not the best to use it to calculate AUC. Will maunally NA this later. 

collect_trace_predictions.py

Convert Trace tool predictions (one or more datasets) to the unified
cell-prediction format used by compute_patient_metrics_all.py.

Trace output columns: clone_id, prediction ('yes'/'no' → saved as pred 1/0), prob_no, prob_yes, confidence

For each dataset two source files are expected in --input_dir:
    <prefix>_sc_predictions.csv      — clone_id == cell barcode (obs_names)
    <prefix>_clone_predictions.csv   — clone_id == sample_clone_id (p75-pooled by Trace)

Study is auto-detected from the filename: any of
    caushi | hanada | lowery | meng | oliveira | zheng | chiffelle
must appear in the filename.

Outputs (both named benchmark_Trace_{study}.csv):
    raw_cell_scores/benchmark_Trace_{study}.csv          ← from sc_predictions
    pooled_scores/p75/benchmark_Trace_{study}.csv        ← from clone_predictions

Output schema (index = cell_barcode):
    score | reactivity | sample_clone_id | sample | pred

Usage
-----
python collect_trace_predictions.py \\
    --input_dir  DIR       directory containing *sc_predictions.csv / *clone_predictions.csv
    [--out_root  DIR]      same root as collect_cell_predictions.py
    [--studies   caushi,chiffelle,...]  restrict to specific studies (default: all found)
    [--h5ad_overrides study=path,...]  override h5ad path for specific studies
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = "/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein"

DEFAULT_OUT_ROOT = (
    f"{BASE}/results/classifier/case_study/"
    "ICB_chiffelle_5case_inference_6AllasTrain/unified_cell_predictions"
)

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

ALL_STUDIES = ["caushi", "hanada", "lowery", "meng", "oliveira", "zheng", "chiffelle", "moffitt"]
OUTPUT_COLS = ["score", "reactivity", "sample_clone_id", "sample", "reactivity_fine", "pred"]

_PRED_MAP = {"yes": 1, "no": 0, "Yes": 1, "No": 0, "YES": 1, "NO": 0, "1":1, "0":0}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_sample_col(obs_df):
    for cname in ["sample", "patient", "batch", "donor", "Sample"]:
        if cname in obs_df.columns:
            return obs_df[cname]
    if "sample_clone_id" in obs_df.columns:
        v0 = obs_df["sample_clone_id"].iloc[0]
        if "_TRA_" in str(v0):
            return obs_df["sample_clone_id"].str.split("_TRA_").str[0]
        return obs_df["sample_clone_id"].str.rsplit("_", n=2).str[0]
    raise ValueError(f"Cannot find sample column. Available: {list(obs_df.columns)}")


def detect_study(filename):
    """Return the first study name found in filename, or None."""
    stem = Path(filename).stem.lower()
    for s in ALL_STUDIES:
        if s in stem:
            return s
    return None


def load_trace_csv(csv_path):
    """Load Trace prediction CSV; return DataFrame indexed by clone_id."""
    df = pd.read_csv(csv_path)
    if "clone_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "clone_id"})
    for col in ("score_i", "prediction"):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in {csv_path}. "
                f"Available: {list(df.columns)}"
            )
    df = df.set_index("clone_id")
    df = df[~df.index.duplicated(keep="first")]
    return df


def _to_reactivity(obs_series):
    return obs_series.map(
        lambda x: int(float(x)) if pd.notna(x) and str(x).strip() not in ("", "nan") else np.nan
    )


# ── Per-study extractors ───────────────────────────────────────────────────────

def extract_sc_level(sc_csv, obs, exclude_sample_list):
    """clone_id == cell barcode. Keep only cells present in prediction file."""
    trace  = load_trace_csv(sc_csv)
    shared = trace.index.intersection(obs.index)

    if len(shared) == 0:
        raise ValueError(
            "No overlap between sc_predictions clone_id and h5ad obs_names.\n"
            f"  First 5 pred IDs  : {trace.index[:5].tolist()}\n"
            f"  First 5 obs_names : {obs.index[:5].tolist()}"
        )

    print(f"    h5ad={len(obs)} | predictions={len(trace)} | matched={len(shared)} "
          f"({len(shared)/len(obs):.1%} of h5ad, {len(shared)/len(trace):.1%} of predictions)")

    trace_sub = trace.loc[shared]
    obs_sub   = obs.loc[shared]

    df = pd.DataFrame(index=shared)
    df.index.name = "cell_barcode"
    si = pd.to_numeric(trace_sub["score_i"], errors="coerce")
    sni = pd.to_numeric(trace_sub["score_ni"], errors="coerce")
    df["score"] = np.sqrt(si**2 + sni**2)
    df["reactivity"]      = _to_reactivity(obs_sub["reactivity"])
    df["sample_clone_id"] = obs_sub["sample_clone_id"].values if "sample_clone_id" in obs_sub.columns else np.nan
    df["reactivity_fine"] = obs_sub["reactivity_fine"].astype(int)
    df["sample"]          = get_sample_col(obs_sub).values
    df["pred"]            = trace_sub["prediction"]

    df = df.dropna(subset=["score", "reactivity", "sample"])
    df["reactivity"] = df["reactivity"].astype(int)
    df = df[~df['sample'].isin(exclude_sample_list)]
    return df[OUTPUT_COLS]


def extract_clone_level(clone_csv, obs, exclude_sample_list):
    """clone_id == sample_clone_id. Attach clone score to each matched cell."""
    trace = load_trace_csv(clone_csv)
    shared = trace.index.intersection(obs.index)

    if "sample_clone_id" not in obs.columns:
        raise ValueError("h5ad obs does not have 'sample_clone_id' column.")

    clone_ids = obs["sample_clone_id"].dropna()
    print(f"    h5ad={len(obs)} | predictions={len(trace)} | matched={len(shared)} "
          f"({len(shared)/len(obs):.1%} of h5ad, {len(shared)/len(trace):.1%} of predictions)")

    trace_sub = trace.loc[shared]
    obs_sub   = obs.loc[shared]


    df = pd.DataFrame(index=shared)
    df.index.name = "cell_barcode"
    si = pd.to_numeric(trace_sub["score_i"], errors="coerce")
    sni = pd.to_numeric(trace_sub["score_ni"], errors="coerce")
    df["score"] = np.sqrt(si**2 + sni**2)
    df["reactivity"]      = _to_reactivity(obs_sub["reactivity"])
    df["sample_clone_id"] = obs_sub["sample_clone_id"].values if "sample_clone_id" in obs_sub.columns else np.nan
    df["reactivity_fine"] = obs_sub["reactivity_fine"].astype(int)
    df["sample"]          = get_sample_col(obs_sub).values
    df["pred"]            = trace_sub["prediction"]

    df = df.dropna(subset=["score", "reactivity", "sample"])
    df["reactivity"] = df["reactivity"].astype(int)
    df = df[~df['sample'].isin(exclude_sample_list)]
    return df[OUTPUT_COLS]


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_study_pairs(input_dir, restrict_studies=None):
    """
    Scan input_dir for *sc_predictions.csv files.
    For each, find the matching *clone_predictions.csv.
    Returns list of (study, sc_path, clone_path).
    """
    input_dir = Path(input_dir)
    pairs = []
    for sc_path in sorted(input_dir.glob("*sc_predictions.csv")):
        study = detect_study(sc_path.name)
        if study is None:
            print(f"  [skip] cannot detect study in filename: {sc_path.name}")
            continue
        if restrict_studies and study not in restrict_studies:
            continue
        # Find clone predictions: same stem with sc_ → clone_
        clone_stem  = sc_path.stem.replace("sc_predictions", "clone_predictions")
        clone_path  = input_dir / f"{clone_stem}.csv"
        if not clone_path.exists():
            # Fallback: glob for any *clone_predictions* with same study
            matches = list(input_dir.glob(f"*{study}*clone_predictions.csv"))
            clone_path = matches[0] if matches else None
        pairs.append((study, sc_path, clone_path))
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Trace predictions (multi-dataset) to unified format"
    )
    parser.add_argument("--input_dir",      required=True,
                        help="Directory containing *sc_predictions.csv / *clone_predictions.csv")
    parser.add_argument("--out_root",       default=DEFAULT_OUT_ROOT,
                        help="Output root directory (same as collect_cell_predictions.py)")
    parser.add_argument("--studies",        default=None,
                        help="Comma-separated study names to process (default: all detected)")
    parser.add_argument("--h5ad_overrides", default=None,
                        help="Override h5ad paths: study=path,study2=path2,... "
                             "(use when h5ad is not in the default LOG1P_H5AD dict)")
    args = parser.parse_args()

    restrict = [s.strip() for s in args.studies.split(",")] if args.studies else None

    # Override h5ad paths if requested
    h5ad_paths = dict(LOG1P_H5AD)
    if args.h5ad_overrides:
        for item in args.h5ad_overrides.split(","):
            study, path = item.split("=", 1)
            h5ad_paths[study.strip()] = path.strip()

    # Output dirs
    out_root = Path(args.out_root)
    raw_dir  = out_root / "raw_cell_scores"
    # since it is going to combine with the max of scTRP, although TRACE use p75, to simplify the ploting process, use max 
    p75_dir  = out_root / "pooled_scores" / "max"
    raw_dir.mkdir(parents=True, exist_ok=True)
    p75_dir.mkdir(parents=True, exist_ok=True)

    # Discover study pairs
    pairs = discover_study_pairs(args.input_dir, restrict)
    if not pairs:
        print(f"No *sc_predictions.csv files found in {args.input_dir}")
        return

    print(f"\nFound {len(pairs)} study pair(s) in {args.input_dir}")

    import scanpy as sc

    results = {}
    
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
    for study, sc_path, clone_path in pairs:
        print(f"\n{'='*65}")
        print(f"Study: {study}")
        print(f"{'='*65}")

        h5ad_path = h5ad_paths.get(study)
        if not h5ad_path or not Path(h5ad_path).exists():
            print(f"  [skip] h5ad not found for study={study!r}: {h5ad_path}")
            print(f"         Use --h5ad_overrides {study}=<path> to specify it.")
            continue

        print(f"  Loading h5ad: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path, backed="r")
        obs   = adata.obs.copy()
        if "reactivity" not in obs.columns:
            print(f"  [skip] h5ad obs missing 'reactivity' column.")
            continue

        fname = f"benchmark_Manascore_{study}.csv"

        # ── 1. Cell-level → raw_cell_scores/ ─────────────────────────────────
        print(f"\n  1. Cell-level  ({sc_path.name})")
        try:
            df_sc  = extract_sc_level(sc_path, obs, exclude_sample_list)
            out_sc = raw_dir / fname
            df_sc.to_csv(out_sc)
            n_cl = df_sc["sample_clone_id"].nunique()
            print(f"    → raw_cell_scores/{fname}  [{len(df_sc)} cells | {n_cl} clones]")
            results.setdefault(study, {})["sc"] = len(df_sc)
        except Exception as e:
            print(f"    [ERROR] {e}")

        # ── 2. Clone-level → pooled_scores/p75/ ──────────────────────────────
        if clone_path is None or not Path(clone_path).exists():
            print(f"\n  2. Clone-level  [skip] clone_predictions CSV not found")
        else:
            print(f"\n  2. Clone-level ({clone_path.name})")
            try:
                df_clone  = extract_clone_level(clone_path, obs, exclude_sample_list)
                out_clone = p75_dir / fname
                df_clone.to_csv(out_clone)
                n_cl = df_clone["sample_clone_id"].nunique()
                print(f"    → pooled_scores/max/{fname}  [{len(df_clone)} cells | {n_cl} clones]")
                results.setdefault(study, {})["clone"] = len(df_clone)
            except Exception as e:
                print(f"    [ERROR] {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Summary")
    print(f"{'='*65}")
    for study, info in results.items():
        sc_n    = info.get("sc",    "–")
        clone_n = info.get("clone", "–")
        print(f"  {study:12s}  sc={sc_n}  clone={clone_n}")

    print(f"\nOutput schema (index = cell_barcode):")
    print(f"  score | reactivity | sample_clone_id | sample | pred")
    print(f"\nTo update downstream metrics:")
    print(f"  python compute_patient_metrics_all.py --in_root {out_root} "
          f"--pool_types raw,p75")


if __name__ == "__main__":
    main()
