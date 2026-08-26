"""
Shared library for the scTRP leave-one-out tutorial (prepare + inference).

Design goal: a third party running the public tutorial notebook never needs the
private per-study train h5ad files. Each model fold ships as:
    {fold_dir}/model_eN.pt        (checkpoint, already distributed)
    {fold_dir}/gene_panel.json    (ordered gene list the model expects)
    {fold_dir}/train_embeddings.npz  (cell_barcode, reactivity, emb — no expression matrix)
    {fold_dir}/val_embeddings.npz    (same, for the held-out validation split)

The last three are produced once, privately, by prepare_fold_embeddings.py.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import jenkspy
import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.insert(0, "/fs/ess/PCON0022/lsxgf/tools_related/scGPT/")
sys.path.insert(0, "/cluster/pixstor/xudong-lab/suli/tools_related/scGPT/")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from torch import nn

from infer_supcon_functions import (
    get_project_emb,
    get_cluster_center,
    knn_classifier,
    nearest_center,
    cosine_similarity_classifier,
    OT_based_prediction,
)

SCGPT_BC_DIR = "/cluster/pixstor/xudong-lab/suli/tools_related/scgpt_data_model/scGPT_bc"
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PAD_VALUE = -2  # input_emb_style == "continuous"
SAMPLE_COL_CANDIDATES = ["sample", "patient", "batch", "donor", "Sample"]
POOL_TYPES = ["raw", "mean", "max", "median", "p75"]


class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, drop_out=0.1):
        super().__init__()
        self.fc1 = nn.Linear(512, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        return self.fc3(x)


def build_config(batch_size=32):
    """Identical hyperparameters to infer_supcon_1004.py's hyperparameter_defaults."""
    return SimpleNamespace(
        seed=0,
        hidden_dim=256,
        out_dim=128,
        do_train=True,
        mask_ratio=0.0,
        n_bins=51,
        MVC=False,
        ecs_thres=0.0,
        dab_weight=0.0,
        lr=1e-4,
        batch_size=batch_size,
        layer_size=128,
        nlayers=4,
        nlayers_cls=3,
        nhead=4,
        dropout=0.2,
        schedule_ratio=0.9,
        fast_transformer=True,
        fast_transformer_backend="flash",
        pre_norm=False,
        amp=True,
        include_zero_gene=False,
        freeze=True,
        DSBN=False,
        input_layer_key="X_binned",
    )


def load_vocab_and_model_configs(scgpt_bc_dir=SCGPT_BC_DIR):
    model_config_file = Path(scgpt_bc_dir) / "args.json"
    vocab_file = Path(scgpt_bc_dir) / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[PAD_TOKEN])
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    return vocab, model_configs


def build_preprocessor(config):
    return Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        result_normed_key="X",
        log1p=False,
        result_log1p_key="X",
        subset_hvg=False,
        hvg_flavor="cell_ranger",
        binning=config.n_bins,
        result_binned_key="X_binned",
    )


def filter_genes_in_vocab(adata, vocab):
    adata = adata.copy()
    adata.var["gene_name"] = adata.var.index.tolist()
    adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in adata.var["gene_name"]]
    matched = int((adata.var["id_in_vocab"] >= 0).sum())
    print(f"match {matched}/{adata.shape[1]} genes in vocabulary of size {len(vocab)}.")
    return adata[:, adata.var["id_in_vocab"] >= 0].copy()


def ini_model(model_file, vocab, model_configs, config, num_types=2, device="cpu"):
    model = TransformerModel(
        len(vocab),
        model_configs["embsize"],
        model_configs["nheads"],
        model_configs["d_hid"],
        model_configs["nlayers"],
        nlayers_cls=config.nlayers_cls,
        n_cls=num_types,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        do_mvc=config.MVC,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=None,
        domain_spec_batchnorm=config.DSBN,
        input_emb_style="continuous",
        n_input_bins=config.n_bins,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=False,
        use_fast_transformer=config.fast_transformer,
        fast_transformer_backend=config.fast_transformer_backend,
        pre_norm=config.pre_norm,
    )
    model.cls_decoder = LayerNormNet(config.hidden_dim, config.out_dim)

    ckpt = torch.load(model_file, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict1", ckpt.get("state_dict", ckpt))
        epoch = ckpt.get("epoch")
    else:
        state_dict, epoch = ckpt, None

    try:
        model.load_state_dict(state_dict)
    except Exception:
        model_dict = model.state_dict()
        matched = {k: v for k, v in state_dict.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)

    if config.freeze:
        for _, p in model.named_parameters():
            p.requires_grad = False
    return model, epoch


def align_test_adata_to_panel(test_adata, gene_panel):
    """Reindex test_adata to exactly the training gene panel (order matters),
    zero-filling any panel gene the test data lacks — without needing the
    private train adata itself, only its saved gene name list."""
    scaffold = ad.AnnData(
        X=np.zeros((1, len(gene_panel)), dtype=np.float32),
        var=pd.DataFrame(index=list(gene_panel)),
    )
    scaffold.obs_names = ["__panel_scaffold__"]
    merged = ad.concat([scaffold, test_adata], join="outer", fill_value=0)
    aligned = merged[list(test_adata.obs_names), list(gene_panel)].copy()
    aligned.var["gene_name"] = aligned.var.index.tolist()
    return aligned


def make_embedding_adata(emb, reactivity_labels):
    """Wrap a bare embedding matrix + labels in a minimal AnnData so we can
    reuse get_cluster_center() etc. unmodified. X is a dummy placeholder —
    no expression data is stored or required here."""
    n = emb.shape[0]
    a = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    a.obs["reactivity"] = np.asarray(reactivity_labels).astype(str)
    a.obsm["X_scGPT_prj"] = np.asarray(emb)
    return a


def distance_based_prediction_with_toN(val_embeddings_np, class_centers_dic, val_labels_np,
                                        actual_val_emb_np, actual_val_labels_np):
    """Verbatim copy of the local override in infer_supcon_1004.py (the version
    with '...toN' negative-center variants), since that shadows the simpler
    one in infer_supcon_functions.py and is what final_infering() actually uses."""

    def _score(emb, center_pos, center_neg=None):
        pos, neg = [], []
        for i in range(emb.shape[0]):
            e = emb[i]
            pos.append(1 - np.linalg.norm(e - center_pos) / 2)
            if center_neg is not None:
                neg.append(np.linalg.norm(e - center_neg) / 2)
        return np.array(pos), (np.array(neg) if center_neg is not None else None)

    from sklearn.metrics import roc_curve

    score_pos, score_neg = _score(actual_val_emb_np, class_centers_dic["1"], class_centers_dic["0"])
    fpr, tpr, thresholds = roc_curve(actual_val_labels_np, score_pos, pos_label="1")
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    fpr, tpr, thresholds = roc_curve(actual_val_labels_np, score_neg, pos_label="1")
    optimal_threshold_toN = thresholds[np.argmax(tpr - fpr)]

    score_pos, score_neg = _score(val_embeddings_np, class_centers_dic["1"], class_centers_dic["0"])

    predictions_Youdensj = np.where(score_pos > optimal_threshold, "1", "0").astype("object")
    breaks = jenkspy.jenks_breaks(score_pos.tolist(), n_classes=2)
    predictions_jenks = np.where(score_pos > breaks[1], "1", "0").astype("object")

    predictions_Youdensj_toN = np.where(score_neg > optimal_threshold_toN, "1", "0").astype("object")
    breaks_toN = jenkspy.jenks_breaks(score_neg.tolist(), n_classes=2)
    predictions_jenks_toN = np.where(score_neg > breaks_toN[1], "1", "0").astype("object")

    return (predictions_Youdensj, predictions_jenks, score_pos,
            predictions_Youdensj_toN, predictions_jenks_toN, score_neg)


def run_fold_inference(model, config, max_seq_len, test_adata, vocab,
                        train_emb, train_labels, val_emb, val_labels, device):
    """Encode `test_adata` with this fold's model and run the full battery of
    KNN / nearest-center / distance / OT classifiers against the fold's
    precomputed train & val embeddings. Returns a per-cell score/pred DataFrame.
    `test_adata` must already be gene-panel-aligned and preprocessed (binned)."""
    genes = test_adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    test_data_ = get_project_emb(
        config, model, test_adata, gene_ids, max_seq_len, vocab,
        PAD_TOKEN, PAD_VALUE, config.include_zero_gene, device,
    )
    obsm_key = "X_scGPT_prj"
    test_embeddings_np = test_data_.obsm[obsm_key]

    train_ad = make_embedding_adata(train_emb, train_labels)
    class_centers = get_cluster_center(train_ad, obsm_key, False)

    train_labels_np = np.asarray(train_labels).astype(str)
    val_labels_np_dummy = np.zeros(test_embeddings_np.shape[0], dtype=object)  # test data usually unlabeled

    predictions, function_names = [], []
    for n in range(1, 11):
        predictions.append(knn_classifier(n, train_emb, train_labels_np, test_embeddings_np))
    function_names += [f"pro_knn_{n}" for n in range(1, 11)]

    nc_pred, nc_score, nc_score_toP = nearest_center(test_embeddings_np, class_centers)
    predictions += [(nc_pred, nc_score), (nc_pred, nc_score_toP)]
    function_names += ["pro_nearest_center", "pro_nearest_center_toP"]

    predictions.append(cosine_similarity_classifier(
        train_emb, test_embeddings_np, train_labels_np, list(pd.unique(train_labels_np))
    ))
    function_names.append("pro_cosine_similarity")

    pro_Y, pro_J, pro_score, pro_Y_toN, pro_J_toN, pro_score_toN = distance_based_prediction_with_toN(
        test_embeddings_np, class_centers, val_labels_np_dummy,
        np.asarray(val_emb), np.asarray(val_labels).astype(str),
    )
    predictions += [(pro_Y, pro_score), (pro_J, pro_score),
                    (pro_Y_toN, pro_score_toN), (pro_J_toN, pro_score_toN)]
    function_names += ["pro_distance_Youdensj", "pro_distance_jenks",
                        "pro_distance_Youdensj_toN", "pro_distance_jenks_toN"]

    ot_predictions, ot_names = OT_based_prediction(train_emb, test_embeddings_np, train_labels_np)
    predictions += ot_predictions
    function_names += [f"pro_{n}" for n in ot_names]

    assert len(predictions) == len(function_names)

    out = pd.DataFrame(index=test_adata.obs_names)
    out.index.name = "cell_barcode"
    for name, (pred, score) in zip(function_names, predictions):
        out[f"{name}_score"] = score
        out[f"{name}_pred"] = pred
    return out


# ── Rank-min ensemble across folds (same logic as collect_cell_predictions.py) ──

def get_sample_col(obs_df):
    for cname in SAMPLE_COL_CANDIDATES:
        if cname in obs_df.columns:
            return obs_df[cname]
    if "sample_clone_id" in obs_df.columns:
        v0 = obs_df["sample_clone_id"].iloc[0]
        if "_TRA_" in v0:
            return obs_df["sample_clone_id"].str.split("_TRA_").str[0]
        return obs_df["sample_clone_id"].str.rsplit("_", n=2).str[0]
    raise ValueError(f"Cannot find sample column. Available: {list(obs_df.columns)}")


def _jenks_thresh_simple(vals):
    try:
        return jenkspy.jenks_breaks(list(vals), n_classes=2)[1]
    except Exception:
        return float(np.median(vals))


def _pool_score_matrix(score_mat, sample_clone_ids, pool_type):
    """Pool each fold's score column by clonotype for one pool_type."""
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


def compute_rankmin_study(score_mat_pooled, sample_s):
    """Rank-min consensus score + per-sample Jenks binary prediction.

    score_s : each cell's minimum study-wide rank across all fold columns
              (worst-case / consensus rank — a cell must rank high in every
              fold to get a high rank_min).
    pred_s  : 1 if score_s exceeds that sample's own Jenks(k=2) threshold.
    """
    rank_mat = score_mat_pooled.rank(method="average", ascending=True, axis=0))
    score_s = rank_mat.min(axis=1).rename("rank_min_study")

    pred_s = pd.Series(0, index=score_s.index, name="pred", dtype=int)
    for samp, idx in sample_s.groupby(sample_s).groups.items():
        scores = score_s.loc[idx].dropna().values.astype(float)
        if len(scores) < 2 or np.unique(scores).size < 2:
            continue
        try:
            thresh = _jenks_thresh_simple(scores)
            pred_s.loc[idx] = (score_s.loc[idx].values > thresh).astype(int)
        except Exception as e:
            print(f"  [jenks] skip sample '{samp}': {e}")
    return score_s, pred_s
