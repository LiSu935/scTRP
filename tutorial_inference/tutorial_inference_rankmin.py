#!/usr/bin/env python3
"""
tutorial_inference_rankmin.py — script version of tutorial_inference_rankmin.ipynb.

Runs a new, external test dataset through the 6 leave-one-out scTRP model folds
under ${ROOT_DIR}/model_folds/{study}/{model_family}/ and combines the 6
per-fold reactivity scores into one final binary call per cell via a rank-min
consensus (a cell must rank high in every fold to be called reactive).

Privacy note: no private train h5ad is used here. Each fold ships as
model_eN.pt + gene_panel.json + train_embeddings.npz + val_embeddings.npz
(produced once, privately, by prepare_fold_embeddings.py) — only projector
embeddings and reactivity labels, never expression values.

Usage:
    python tutorial_inference_rankmin.py \\
        --root_dir /path/to/root_dir \\
        --test_h5ad_path /path/to/your_test_data.h5ad

Output: rank_min_final_predictions.csv, written next to --test_h5ad_path.
"""
import scanpy as sc
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tutorial_infer_functions import (
    build_config,
    build_preprocessor,
    load_vocab_and_model_configs,
    ini_model,
    align_test_adata_to_panel,
    run_fold_inference,
    get_sample_col,
    _pool_score_matrix,
    compute_rankmin_study,
    POOL_TYPES,
    SCGPT_BC_DIR,
)

SIX_STUDIES = ["caushi", "hanada", "lowery", "meng", "oliveira", "zheng"]
SCORE_COL = "pro_jenks_OT_deltarho_30best_score"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True,
                    help="Contains model_folds/{study}/{model_family}/ per fold")
    p.add_argument("--test_h5ad_path", required=True,
                    help="Your test dataset; output CSV is written alongside it")
    p.add_argument("--model_family", default="scTRP_simclr",
                    choices=["scTRP_simclr", "scTRP_only"])
    p.add_argument("--studies", nargs="+", default=SIX_STUDIES, choices=SIX_STUDIES)
    p.add_argument("--scgpt_bc_dir", default=SCGPT_BC_DIR)
    p.add_argument("--max_seq_len", type=int, default=1200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pool_type", default="max", choices=POOL_TYPES,
                    help="Clonotype pooling before the rank-min ensemble; "
                         "'raw' = no pooling")
    p.add_argument("--out_csv", default=None,
                    help="Default: rank_min_final_predictions.csv next to --test_h5ad_path")
    p.add_argument("--score_mat_csv", default=None,
                    help="Default: fold_scores.csv next to --test_h5ad_path")
    return p.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    test_h5ad_path = Path(args.test_h5ad_path)
    fold_dirs = {s: root_dir / "model_folds" / s / args.model_family for s in args.studies}

    out_csv = Path(args.out_csv) if args.out_csv else test_h5ad_path.parent / "rank_min_final_predictions.csv"
    score_mat_csv = Path(args.score_mat_csv) if args.score_mat_csv else test_h5ad_path.parent / "fold_scores.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = build_config(args.batch_size)
    vocab, model_configs = load_vocab_and_model_configs(args.scgpt_bc_dir)
    preprocessor = build_preprocessor(config)

    test_adata_raw = sc.read_h5ad(test_h5ad_path)
    test_adata_raw.var["gene_name"] = test_adata_raw.var.index.tolist()
    print(test_adata_raw)

    fold_scores = {}
    for study, fold_dir in fold_dirs.items():
        print(f"\n=== fold: {study} ===")
        pt_path = sorted(Path(fold_dir).glob("model_e*.pt"))[0]

        with open(Path(fold_dir) / "gene_panel.json") as f:
            gene_panel = json.load(f)
        train_npz = np.load(Path(fold_dir) / "train_embeddings.npz", allow_pickle=True)
        val_npz = np.load(Path(fold_dir) / "val_embeddings.npz", allow_pickle=True)

        test_adata = align_test_adata_to_panel(test_adata_raw, gene_panel)
        preprocessor(test_adata, batch_key=None)

        model, epoch = ini_model(str(pt_path), vocab, model_configs, config, device=device)
        model.to(device)

        fold_df = run_fold_inference(
            model, config, args.max_seq_len, test_adata, vocab,
            train_emb=train_npz["emb"], train_labels=train_npz["reactivity"],
            val_emb=val_npz["emb"], val_labels=val_npz["reactivity"],
            device=device,
        )
        fold_scores[study] = fold_df[SCORE_COL]

        del model
        torch.cuda.empty_cache()

    score_mat = pd.DataFrame(fold_scores)
    score_mat.to_csv(score_mat_csv)
    print(f"wrote {score_mat_csv}")

    sample_s = get_sample_col(test_adata_raw.obs).reindex(score_mat.index)
    sample_clone_ids = (
        test_adata_raw.obs["sample_clone_id"].reindex(score_mat.index)
        if "sample_clone_id" in test_adata_raw.obs.columns else None
    )

    score_mat_pooled = _pool_score_matrix(score_mat, sample_clone_ids, args.pool_type)
    rank_min_score, rank_min_pred = compute_rankmin_study(score_mat_pooled, sample_s)

    final = score_mat.copy()
    final["sample"] = sample_s
    if sample_clone_ids is not None:
        final["sample_clone_id"] = sample_clone_ids
    final["rank_min_study"] = rank_min_score
    final["pred"] = rank_min_pred

    print(final["pred"].value_counts())
    final.to_csv(out_csv)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
