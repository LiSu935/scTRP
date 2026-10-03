#!/usr/bin/env python3
"""
Step 2 — Tokenize a log1p .h5ad with the scGPT vocabulary and write one .npz per cell.

Each cell is written to {output_dir}/{barcode}-{prefix}.npz with keys:
    index, genes_data, expressions_data, masked_expr_data, seq
where seq = adata.obs['cdr3_aa'] and prefix = --prefix (default: the h5ad file stem).
Genes that are not in the scGPT vocabulary are dropped.

Usage:
    python tokenize_h5ad.py \\
        --input_h5ad      /path/to/sample_log1p.h5ad \\
        --output_dir      /path/to/work/sample/tokenized_data/ \\
        --scgpt_model_dir /path/to/scgpt_human \\
        --prefix          sample
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from torch.utils.data import DataLoader, SequentialSampler

SPECIAL_TOKENS = ["<pad>", "<cls>", "<eoc>"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_h5ad", required=True, help="Output of prep_log1p_h5ad.py")
    p.add_argument("--output_dir", required=True, help="Directory for the per-cell .npz files")
    p.add_argument("--scgpt_model_dir", required=True,
                   help="Dir containing vocab.json, args.json and best_model.pt of the scGPT model")
    p.add_argument("--scgpt_path", default=None,
                   help="Optional path to the scGPT source tree, if scgpt is not pip-installed")
    p.add_argument("--prefix", default=None,
                   help="Suffix appended to every cell id, e.g. the sample name. Default: h5ad file stem")
    p.add_argument("--gene_col", default="Gene Symbol",
                   help="adata.var column with gene symbols. Default 'Gene Symbol' "
                        "(falls back to var_names if the column is missing)")
    p.add_argument("--max_length", type=int, default=1200, help="Max tokens per cell. Default 1200")
    return p.parse_args()


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, X, gene_ids, cls_id, pad_value):
        self.X = X
        self.gene_ids = gene_ids
        self.cls_id = cls_id
        self.pad_value = pad_value

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        nonzero_idx = np.nonzero(row)[0]
        genes = np.insert(self.gene_ids[nonzero_idx], 0, self.cls_id)
        values = np.insert(row[nonzero_idx], 0, self.pad_value)
        return {"id": idx, "genes": torch.from_numpy(genes).long(), "expressions": torch.from_numpy(values)}


def to_numpy(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def main():
    args = parse_args()
    if args.scgpt_path:
        sys.path.insert(0, args.scgpt_path)
    from scgpt.data_collator import DataCollator
    from scgpt.tokenizer import GeneVocab

    model_dir = Path(args.scgpt_model_dir)
    with open(model_dir / "args.json") as f:
        model_configs = json.load(f)

    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    adata = sc.read_h5ad(args.input_h5ad)
    if args.gene_col not in adata.var.columns:
        adata.var[args.gene_col] = adata.var_names.tolist()
    prefix = args.prefix or Path(args.input_h5ad).stem

    # keep only genes present in the scGPT vocabulary
    adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata.var[args.gene_col]]
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
    print(f"{adata.n_vars} genes in vocabulary; {adata.n_obs} cells")

    gene_ids = np.array(adata.var["id_in_vocab"], dtype=int)
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    seqs = adata.obs["cdr3_aa"].astype(str).tolist()

    dataset = CellDataset(X, gene_ids, cls_id=vocab["<cls>"], pad_value=model_configs["pad_value"])
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=args.max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                        collate_fn=collator, drop_last=False)

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    for batch in loader:
        # batch_size=1, so batch[...][0] is the single cell at position `count`
        cell_id = f"{adata.obs_names[count]}-{prefix}"  # makes the id unique across samples
        np.savez(
            os.path.join(args.output_dir, f"{cell_id}.npz"),
            index=cell_id,
            genes_data=to_numpy(batch["gene"][0]),
            expressions_data=to_numpy(batch["expr"][0]),
            masked_expr_data=to_numpy(batch["masked_expr"][0]),
            seq=seqs[count],
        )
        count += 1

    assert count == adata.n_obs, f"wrote {count} npz files for {adata.n_obs} cells"
    print(f"wrote {count} npz files to {args.output_dir}")


if __name__ == "__main__":
    main()
