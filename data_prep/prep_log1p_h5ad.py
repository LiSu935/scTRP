#!/usr/bin/env python3
"""
Step 1 — Build a log1p AnnData (.h5ad) for training from a 10x output directory.

Runs QC, TCR contig filtering, CPM-normalisation + log1p and HVG selection through
load_TCR_RNA_prep.prep_TCR_RNA, then attaches:
    adata.obs['cdr3_aa']     VDJ_1 CDR3 amino-acid sequence (used by the ESM2 encoder)
    adata.obs['reactivity']  binary label '0' / '1' from --reactivity_csv (REQUIRED)
    adata.var['Gene Symbol'] gene names (used by the scGPT tokenizer)

Cells without a reactivity label or a CDR3 sequence are dropped.

Usage:
    python prep_log1p_h5ad.py \\
        --input_dir      /path/to/10x_dir/ \\
        --reactivity_csv /path/to/reactivity.csv \\
        --output_h5ad    /path/to/out/sample_log1p.h5ad

--reactivity_csv: a CSV whose index is the cell barcode (e.g. AAACCTGAGAAGGCCT-1) and
which has a column named 'reactivity' holding 0 or 1.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import scirpy as ir

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_TCR_RNA_prep as prep


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True,
                   help="10x output dir with *feature_bc_matrix.h5 and *_t_*_contig_annotations.csv")
    p.add_argument("--reactivity_csv", required=True,
                   help="CSV indexed by cell barcode with a 'reactivity' column (0/1). Required.")
    p.add_argument("--output_h5ad", required=True)
    p.add_argument("--n_genes_cut", type=int, default=5000,
                   help="Drop cells with n_genes_by_counts >= this. Default 5000.")
    p.add_argument("--pct_mt_cut", type=int, default=20,
                   help="Drop cells with pct_counts_mt >= this. Default 20.")
    p.add_argument("--gene_list", default=None,
                   help="Optional CSV (first column = gene symbols). If omitted, 2000 HVGs are used.")
    return p.parse_args()


def main():
    args = parse_args()

    # prep_TCR_RNA concatenates file patterns onto input_dir, so it needs a trailing slash
    input_dir = os.path.join(args.input_dir, "")

    gene_list = None
    if args.gene_list:
        gene_list = pd.read_csv(args.gene_list, header=None).iloc[:, 0].tolist()

    labels = pd.read_csv(args.reactivity_csv, index_col=0)
    if "reactivity" not in labels.columns:
        raise ValueError(f"{args.reactivity_csv} must have a 'reactivity' column (0/1); "
                         f"found columns: {list(labels.columns)}")

    mdata = prep.prep_TCR_RNA(
        input_dir,
        n_genes_cut=args.n_genes_cut,
        pct_mt_cut=args.pct_mt_cut,
        use_gene_subsets=gene_list is not None,
        geneList=gene_list,
    )

    adata = mdata["gex"].copy()
    cdr3 = ir.get.airr(mdata["airr"], ["cdr3_aa"], ("VDJ_1",))["VDJ_1_cdr3_aa"]

    adata.obs["cdr3_aa"] = cdr3.reindex(adata.obs_names).values
    reactivity = labels["reactivity"].reindex(adata.obs_names)
    print(f"cells after QC: {adata.n_obs}; with reactivity label: {reactivity.notna().sum()}")

    keep = reactivity.notna() & adata.obs["cdr3_aa"].notna()
    adata = adata[keep.values].copy()
    adata.obs["reactivity"] = reactivity[keep].astype(int).astype(str).values
    adata.var["Gene Symbol"] = adata.var_names

    print(f"kept {adata.n_obs} labelled cells with CDR3; label counts:")
    print(adata.obs["reactivity"].value_counts())

    os.makedirs(os.path.dirname(os.path.abspath(args.output_h5ad)), exist_ok=True)
    adata.write_h5ad(args.output_h5ad)
    print(f"wrote {args.output_h5ad}")


if __name__ == "__main__":
    main()
