# scripts/infer.py
#
# Post-training inference: load one or more checkpoint .pt files, encode
# train and test data with the scGPT model, then predict reactivity using
# KNN / nearest-center / cosine / distance / OT classifiers.
#
# EXAMPLE COMMAND:
#   python scripts/infer.py \
#       --load_local_pretrain_pathList /path/to/model_e24.pt,/path/to/model_e36.pt \
#       --train_data_path /path/to/train.h5ad \
#       --test_data_path  /path/to/test.h5ad
#
# ARGUMENTS:
#   --load_local_pretrain_pathList  Comma-separated list of checkpoint .pt paths
#   --train_data_path               Path to the training .h5ad (reactivity labels required)
#   --test_data_path                Path to the test .h5ad to classify
#   --test_new_data                 Flag: set if test data is from a new cohort
#   --max_seq_len                   scGPT token sequence length (default 1200)
#   --batch_size                    Batch size for embedding (default 32)

import os
import glob

from numba import config
config.DISABLE_CACHING = True

# Pre-parse --scgpt_path from argv so it can be applied before any scgpt imports.
import sys as _sys
_idx = next((i for i, a in enumerate(_sys.argv) if a == '--scgpt_path'), None)
if _idx is not None and _idx + 1 < len(_sys.argv):
    os.environ.setdefault('SCGPT_PATH', _sys.argv[_idx + 1])

from types import SimpleNamespace
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch import nn
from torch.nn import functional as F
import anndata as ad
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# ---- scGPT source path ----
# Priority: --scgpt_path arg (pre-parsed above)  →  SCGPT_PATH env var  →  known paths.
# Known fallback paths (examples — pass --scgpt_path to override):
#   /fs/ess/PCON0022/lsxgf/tools_related/scGPT/
#   /cluster/pixstor/xudong-lab/suli/tools_related/scGPT/
_env_scgpt = os.environ.get("SCGPT_PATH")
if _env_scgpt:
    sys.path.insert(0, _env_scgpt)
else:
    for _p in [
        "/fs/ess/PCON0022/lsxgf/tools_related/scGPT/",
        "/cluster/pixstor/xudong-lab/suli/tools_related/scGPT/",
    ]:
        if Path(_p).exists():
            sys.path.insert(0, _p)
            break

import argparse
import json
import jenkspy
import flash_attn
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
from sklearn.metrics import roc_auc_score

# ---- scTRP package imports ----
from scTRP.inference.functions import (
    get_project_emb, get_cluster_center,
    knn_classifier, nearest_center, cosine_similarity_classifier,
    OT_based_prediction, output_metrics,
)
from scTRP.models.layers import LayerNormNet

from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

os.environ["KMP_WARNINGS"] = "off"


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_local_pretrain_pathList', type=str, default=None,
                        help='Comma-separated list of checkpoint .pt paths')
    parser.add_argument('--max_seq_len', default=1200, type=int)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--test_new_data', action='store_true')
    # ---- scGPT paths ----
    parser.add_argument('--scgpt_path', type=str, default=None,
                        help='Path to scGPT source directory (for import). '
                             'Alternative: set SCGPT_PATH env var. '
                             'Examples: /fs/ess/PCON0022/lsxgf/tools_related/scGPT/ '
                             'or /cluster/pixstor/xudong-lab/suli/tools_related/scGPT/')
    parser.add_argument('--scgpt_model_path', type=str, default=None,
                        help='Path to the scGPT pretrained model directory (contains '
                             'args.json, vocab.json, best_model.pt). '
                             'Examples: /cluster/pixstor/xudong-lab/suli/tools_related/scgpt_data_model/scGPT_bc '
                             'or /fs/ess/PCON0022/lsxgf/tools_related/scgpt_data_model/scGPT_bc')
    args = parser.parse_args()
    return args


args = parse()

model_file_list = [n for n in args.load_local_pretrain_pathList.split(',')]
max_seq_len = args.max_seq_len
batch_size  = args.batch_size
train_data_path = args.train_data_path
test_data_path  = args.test_data_path
test_new_data   = args.test_new_data
test_file_name_noextension = os.path.splitext(os.path.basename(args.test_data_path))[0]

# ---- Model hyperparameters (fixed for inference) ----
hyperparameter_defaults = dict(
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
    save_eval_interval=10,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=True,
    DSBN=False,
)
config = SimpleNamespace(**hyperparameter_defaults)
set_seed(config.seed)

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"
include_zero_gene = config.include_zero_gene
n_bins = config.n_bins

MLM = False; CLS = True; ADV = False; CCE = False
MVC = config.MVC; ECS = config.ecs_thres > 0
DAB = False; INPUT_BATCH_LABELS = False
input_emb_style = "continuous"
cell_emb_style  = "cls"
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight    = config.dab_weight
num_batch_types = None

explicit_zero_prob  = MLM and include_zero_gene
do_sample_in_train  = False and explicit_zero_prob
per_seq_batch_sample = False

fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"
embsize = config.layer_size
d_hid   = config.layer_size
nlayers = config.nlayers
nhead   = config.nhead
dropout = config.dropout

mask_value  = -1
pad_value   = -2
n_input_bins = n_bins

DAB_separate_optim = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32

# ---- scGPT pretrained model directory ----
# Pass --scgpt_model_path to override. Known fallback paths (examples):
#   /cluster/pixstor/xudong-lab/suli/tools_related/scgpt_data_model/scGPT_bc
#   /fs/ess/PCON0022/lsxgf/tools_related/scgpt_data_model/scGPT_bc
if args.scgpt_model_path is not None:
    scgpt_model_path = args.scgpt_model_path
elif Path('/cluster/pixstor/xudong-lab/suli/').exists():
    scgpt_model_path = '/cluster/pixstor/xudong-lab/suli/tools_related/scgpt_data_model/scGPT_bc'
elif Path('/fs/ess/PCON0022/lsxgf/').exists():
    scgpt_model_path = '/fs/ess/PCON0022/lsxgf/tools_related/scgpt_data_model/scGPT_bc'
else:
    raise RuntimeError(
        "Cannot locate scGPT pretrained model. Pass --scgpt_model_path /path/to/scGPT_bc"
    )

model_config_file = Path(scgpt_model_path) / "args.json"
vocab_file        = Path(scgpt_model_path) / "vocab.json"

# ---- Load data ----
adata = sc.read_h5ad(train_data_path)
adata.var["gene_name"] = adata.var.index.tolist()

test_adata = sc.read_h5ad(test_data_path)
test_adata.var["gene_name"] = test_adata.var.index.tolist()

test_adata = ad.concat([adata, test_adata], join="outer", fill_value=0)[
    list(test_adata.obs_names), adata.var.index.tolist()
]
test_adata.var["gene_name"] = test_adata.var.index.tolist()

num_types = 2
data_is_raw = False

vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}.")
adata = adata[:, adata.var["id_in_vocab"] >= 0]

test_adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in test_adata.var["gene_name"]]
test_adata = test_adata[:, test_adata.var["id_in_vocab"] >= 0]

vocab.set_default_index(vocab["<pad>"])

with open(model_config_file, "r") as f:
    model_configs = json.load(f)
embsize = model_configs["embsize"]
nhead   = model_configs["nheads"]
d_hid   = model_configs["d_hid"]
nlayers = model_configs["nlayers"]

preprocessor = Preprocessor(
    use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
    normalize_total=False, result_normed_key="X", log1p=data_is_raw,
    result_log1p_key="X", subset_hvg=False,
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins, result_binned_key="X_binned",
)
preprocessor(adata, batch_key=None)

train_adata, val_adata_o = train_test_split(adata, test_size=0.1, stratify=adata.obs['reactivity'], random_state=42)
preprocessor(test_adata, batch_key=None)

input_layer_key = "X_binned"
config.input_layer_key = input_layer_key

genes    = adata.var["gene_name"].tolist()
gene_ids = np.array(vocab(genes), dtype=int)
ntokens  = len(vocab)


def distance_based_prediction(val_embeddings_np, class_centers_dic, val_labels_np,
                               actual_val_emb_np, actual_val_labels_np):
    distance_prediction_score = []
    for i in range(actual_val_emb_np.shape[0]):
        distance = np.linalg.norm(actual_val_emb_np[i] - class_centers_dic['1'])
        distance_prediction_score.append(1 - distance / 2)
    distance_prediction_score = np.array(distance_prediction_score)
    fpr, tpr, thresholds = roc_curve(actual_val_labels_np, distance_prediction_score, pos_label='1')
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    distance_prediction_score = []
    distance_prediction_score_toN = []
    for i in range(val_embeddings_np.shape[0]):
        d_pos = np.linalg.norm(val_embeddings_np[i] - class_centers_dic['1'])
        d_neg = np.linalg.norm(val_embeddings_np[i] - class_centers_dic['0'])
        distance_prediction_score.append(1 - d_pos / 2)
        distance_prediction_score_toN.append(d_neg / 2)
    distance_prediction_score   = np.array(distance_prediction_score)
    distance_prediction_score_toN = np.array(distance_prediction_score_toN)

    fpr, tpr, thresholds = roc_curve(actual_val_labels_np, distance_prediction_score_toN, pos_label='1')
    optimal_threshold_toN = thresholds[np.argmax(tpr - fpr)]

    predictions_Youdensj = np.where(distance_prediction_score > optimal_threshold, '1', '0').astype('object')
    df_tem  = pd.DataFrame({'pred_scores': distance_prediction_score})
    breaks  = jenkspy.jenks_breaks(df_tem['pred_scores'], n_classes=2)
    predictions_jenks = np.where(distance_prediction_score > breaks[1], '1', '0').astype('object')

    predictions_Youdensj_toN = np.where(distance_prediction_score_toN > optimal_threshold_toN, '1', '0').astype('object')
    df_tem2 = pd.DataFrame({'pred_scores': distance_prediction_score_toN})
    breaks2 = jenkspy.jenks_breaks(df_tem2['pred_scores'], n_classes=2)
    predictions_jenks_toN = np.where(distance_prediction_score_toN > breaks2[1], '1', '0').astype('object')

    return (predictions_Youdensj, predictions_jenks, distance_prediction_score,
            predictions_Youdensj_toN, predictions_jenks_toN, distance_prediction_score_toN)


def ini_model(model_file):
    model = TransformerModel(
        ntokens, embsize, nhead, d_hid, nlayers,
        nlayers_cls=config.nlayers_cls,
        n_cls=num_types if CLS else 1,
        vocab=vocab, dropout=dropout,
        pad_token=pad_token, pad_value=pad_value,
        do_mvc=MVC, do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=config.DSBN,
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=config.pre_norm,
    )
    model.cls_decoder = LayerNormNet(config.hidden_dim, config.out_dim)
    ckpt = torch.load(model_file, map_location=device)
    if isinstance(ckpt, dict):
        pretrained_state_dic = ckpt.get('state_dict1', ckpt.get('state_dict', ckpt))
    else:
        pretrained_state_dic = ckpt
    epoch = ckpt.get('epoch') if isinstance(ckpt, dict) else None
    try:
        model.load_state_dict(pretrained_state_dic)
    except:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_state_dic.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    for name, para in model.named_parameters():
        if config.freeze:
            para.requires_grad = False
    return model, epoch


def final_infering(save_dir, cfg, mdl, epc, val_data, train_data_, dev, dt, actual_val_data_, out_prefix, ret_loss):
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.amp):
        val_data_ = get_project_emb(cfg, mdl, val_data, gene_ids, max_seq_len, vocab, pad_token, pad_value, include_zero_gene, dev)
        val_adata_output_file = save_dir / f"{out_prefix}_e{epc}-{time.strftime('%b%d-%H-%M')}.h5ad"
        val_data_.write(val_adata_output_file, compression='gzip')

        unique_labels = list(train_data_.obs['reactivity'].unique())
        predictions, function_names = [], []
        pref_, obsm_key = "projector", "X_scGPT_prj"

        class_centers          = get_cluster_center(train_data_, obsm_key, False)
        class_centers_norm_train = get_cluster_center(train_data_, obsm_key, True)

        train_embeddings_np      = train_data_.obsm[obsm_key]
        val_embeddings_np        = val_data_.obsm[obsm_key]
        actual_val_embeddings_np = actual_val_data_.obsm[obsm_key]

        train_labels_np      = train_data_.obs["reactivity"].values
        val_labels_np        = val_data_.obs["reactivity"].values
        actual_val_labels_np = actual_val_data_.obs["reactivity"].values

        class_centers_norm_val = get_cluster_center(actual_val_data_, obsm_key, True)
        pro_class_centers_distance_val = np.linalg.norm(class_centers_norm_val['0'] - class_centers_norm_val['1'])

        for n in range(1, 11):
            predictions.append(knn_classifier(n, train_embeddings_np, train_labels_np, val_embeddings_np))

        nc_predictions, nc_score, nc_score_toP = nearest_center(val_embeddings_np, class_centers)
        predictions.append((nc_predictions, nc_score))
        predictions.append((nc_predictions, nc_score_toP))
        predictions.append(cosine_similarity_classifier(train_embeddings_np, val_embeddings_np, train_labels_np, unique_labels))

        (pY, pJ, dY_score, pY_toN, pJ_toN, dN_score) = distance_based_prediction(
            val_embeddings_np, class_centers, val_labels_np,
            actual_val_embeddings_np, actual_val_labels_np,
        )
        predictions += [(pY, dY_score), (pJ, dY_score), (pY_toN, dN_score), (pJ_toN, dN_score)]

        function_names = (
            [pref_ + '_knn_' + str(n) for n in range(1, 11)] +
            [pref_ + "_nearest_center", pref_ + "_nearest_center_toP", pref_ + "_cosine_similarity",
             pref_ + "_distance_Youdensj", pref_ + "_distance_jenks",
             pref_ + "_distance_Youdensj_toN", pref_ + "_distance_jenks_toN"]
        )

        OT_prediction_list, OT_function_list = OT_based_prediction(train_embeddings_np, val_embeddings_np, train_labels_np)
        predictions    += OT_prediction_list
        function_names += OT_function_list

        assert len(predictions) == len(function_names)
        print(f"len of predictions list is: {len(predictions)}")

        val_data_.obs['pro_knn7_preds']  = predictions[6][0]
        val_data_.obs['pro_nc_preds']    = predictions[10][0]
        val_data_.obs['pro_nc_toP']      = predictions[11][0]
        val_data_.obs['pro_cosine_sim']  = predictions[12][0]
        val_data_.obs['pro_dYoudensj']   = predictions[13][0]
        val_data_.obs['pro_dJenks']      = predictions[14][0]
        val_data_.obs['pro_dYoudenj_toN'] = predictions[15][0]
        val_data_.obs['pro_dJenks_toN']  = predictions[16][0]
        val_data_.obs['pro_top10_OT_KNN']     = predictions[17][0]
        val_data_.obs['pro_jenks_OT_KNN']     = predictions[18][0]
        val_data_.obs['pro_top10_OT_deltarho'] = predictions[19][0]
        val_data_.obs['pro_jenks_OT_deltarho'] = predictions[20][0]

        for i, name in enumerate(function_names):
            val_data_.obs[f'score_{name}'] = predictions[i][1]

        val_data_.write(val_adata_output_file, compression='gzip')

        if len(np.unique(val_labels_np)) > 1:
            output_file = save_dir / f"{out_prefix}_prediction_scores_e{epc}_inferPostTrain.csv"
            output_metrics(output_file, predictions, function_names, val_labels_np)

        print("inference finished!")


# ============================================================ #
# Main inference loop
# ============================================================ #
save_dir_list = [Path(os.path.dirname(pt_path)) for pt_path in model_file_list]
for save_dir in save_dir_list:
    save_dir.mkdir(parents=True, exist_ok=True)

for model_file, save_dir in zip(model_file_list, save_dir_list):
    model, epoch = ini_model(model_file)
    model.to(device)
    basename = os.path.splitext(os.path.basename(model_file))[0]
    if epoch is None:
        try:
            epoch = basename.split('_e')[1]
        except IndexError:
            epoch = None

    if not test_new_data:
        out_prefix = "testing"
        file_pattern = os.path.join(save_dir, f"train_e{epoch}-*.h5ad")
        matching_files = glob.glob(file_pattern)
        if matching_files:
            train_data_ = sc.read_h5ad(matching_files[0])
        else:
            train_data_ = get_project_emb(config, model, train_adata, gene_ids, max_seq_len, vocab, pad_token, pad_value, include_zero_gene, device)
            train_data_.write(save_dir / f"train_e{epoch}-{time.strftime('%b%d-%H-%M')}.h5ad", compression='gzip')
        file_pattern = os.path.join(save_dir, f"val_e{epoch}-*.h5ad")
        matching_files = glob.glob(file_pattern)
        if matching_files:
            actual_val_data_ = sc.read_h5ad(matching_files[0])
        else:
            actual_val_data_ = get_project_emb(config, model, val_adata_o, gene_ids, max_seq_len, vocab, pad_token, pad_value, include_zero_gene, device)
            actual_val_data_.write(save_dir / f"val_e{epoch}-{time.strftime('%b%d-%H-%M')}.h5ad", compression='gzip')
    else:
        out_prefix = test_file_name_noextension
        train_data_ = get_project_emb(config, model, train_adata, gene_ids, max_seq_len, vocab, pad_token, pad_value, include_zero_gene, device)
        train_data_.write(save_dir / f"train_e{epoch}_for_{test_file_name_noextension}-{time.strftime('%b%d-%H-%M')}.h5ad", compression='gzip')
        actual_val_data_ = get_project_emb(config, model, val_adata_o, gene_ids, max_seq_len, vocab, pad_token, pad_value, include_zero_gene, device)
        actual_val_data_.write(save_dir / f"val_e{epoch}_for_{test_file_name_noextension}-{time.strftime('%b%d-%H-%M')}.h5ad", compression='gzip')

    final_infering(save_dir, config, model, epoch, test_adata, train_data_, device, dtype, actual_val_data_, out_prefix, False)
