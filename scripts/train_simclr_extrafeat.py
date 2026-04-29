# scripts/train_simclr_extrafeat.py
#
# SimCLR + SupCon-Hard training with ESM2 embeddings + optional extra features
# (e.g. HMM similarity score).
#
# Key design:
#   - model_gex  : scGPT TransformerModel (frozen encoder + trainable cls_decoder)
#   - model_seq  : MoBYMLP projector (in_dim = 1280 + extra_feat_dim)
#   - Loss       : SupConHardLoss (scGPT CLS embeddings)
#                + SimCLR info-NCE (scGPT ↔ ESM2+extra_feats)
#
# Data prep:
#   python classifier/data_prep/encode_seq_with_pretrained_ESM2_extraFeat.py \
#       --data_url  <train.tar>  --extra_feat_csv <train_feats.csv> \
#       --split train  --norm_stats_json norm_stats.json
#
# EXAMPLE COMMAND:
#   python scripts/train_simclr_extrafeat.py \
#       --train_data_path  /path/to/train.h5ad \
#       --val_data_path    /path/to/val.h5ad \
#       --test_data_path   /path/to/test.h5ad \
#       --train_data_path_simclr  /path/to/train_esm2encoded_extrafeat.tar \
#       --val_data_path_simclr    /path/to/val_esm2encoded_extrafeat.tar \
#       --extra_feat_dim 1 \
#       --lr 1e-4  --lr_seq 1e-4  --epochs 100  --batch_size 32 \
#       --n_pos 9  --n_neg 30  --temp 0.1  --simclr_weight 1.0 \
#       --unfix_last_layer 1  --out_dim 128  --hidden_dim 256 \
#       --model_name MyExperiment_HMM
#
# TUNABLE PARAMETERS:
#   --extra_feat_dim   N extra features appended to ESM2 (default 0 = disabled)
#   --lr               scGPT learning rate (default 1e-4)
#   --lr_seq           MoBYMLP learning rate (default 1e-4)
#   --epochs           number of epochs (default 100)
#   --batch_size       batch size (default 32)
#   --temp             SupCon-Hard temperature (default 0.1)
#   --n_pos            positives per anchor (default 9)
#   --n_neg            negatives per anchor (default 30)
#   --temperature      SimCLR info-NCE temperature (default 0.05)
#   --simclr_weight    weight of SimCLR loss (default 1.0)
#   --unfix_last_layer unfreeze last N scGPT transformer layers (default 0)
#   --out_dim          projector output dim (default 128)
#   --hidden_dim       cls_decoder hidden dim (default 256)
#   --save_eval_interval  checkpoint/eval every N epochs (default 10)

from numba import config
config.DISABLE_CACHING = True

# Pre-parse --scgpt_path from argv so it can be applied before any scgpt imports.
import os as _os, sys as _sys
_idx = next((i for i, a in enumerate(_sys.argv) if a == '--scgpt_path'), None)
if _idx is not None and _idx + 1 < len(_sys.argv):
    _os.environ.setdefault('SCGPT_PATH', _sys.argv[_idx + 1])

import scanpy as sc
import scvi
import numpy as np
import pandas as pd
from anndata import AnnData

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import random

import pickle
import seaborn as sns
import torch

import cosine_annealing_warmup

import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from collections.abc import Sequence
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

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

import flash_attn
from scgpt.tokenizer.gene_tokenizer import GeneVocab

import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int

# ---- scTRP package imports (replaces inline definitions) ----
from scTRP.inference.functions import (
    return_count_data, get_project_emb, get_cluster_center,
    knn_classifier, nearest_center, cosine_similarity_classifier,
    distance_based_prediction, OT_based_prediction, output_metrics,
)
from scTRP.models.layers import LayerNormNet, MoBYMLP
from scTRP.losses import SupConHardLoss
from scTRP.training.simclr import (
    AverageMeter, log_negative_mean_logtis,
    build_seq_input, SimCLRWithExtraFeat,
)

import webdataset as wds
from torch.cuda.amp import autocast

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"


# ============================================================ #
# Argument parsing
# ============================================================ #
def parse():
    parser = argparse.ArgumentParser()

    # ---- data paths ----
    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--train_data_path_simclr', type=str, default=None)
    parser.add_argument('--val_data_path_simclr', type=str, default=None)

    # ---- model architecture ----
    parser.add_argument('-d', '--hidden_dim', type=int, default=256)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--nlayers_cls', type=int, default=3)
    parser.add_argument('--unfix_last_layer', type=int, default=0)
    parser.add_argument('--max_seq_len', default=1200, type=int)
    parser.add_argument('--extra_feat_dim', type=int, default=0,
                        help='Number of extra features appended to the ESM2 embedding. '
                             'Default 0 = disabled (identical to train_simclr.py behaviour).')

    # ---- training ----
    parser.add_argument('-l',   '--lr',       type=float, default=1e-4)
    parser.add_argument('-l_seq', '--lr_seq', type=float, default=1e-4)
    parser.add_argument('-e',   '--epochs',   type=int,   default=100)
    parser.add_argument('-b',   '--batch_size', type=int, default=32)
    parser.add_argument('--optimi', type=str, default="Adam")
    parser.add_argument('--warmup', default=50, type=int)
    parser.add_argument('--first_cycle_steps', default=3162, type=int)
    parser.add_argument('--warmup_gamma', default=1.0, type=float)

    # ---- SupCon-Hard ----
    parser.add_argument('-T', '--temp', type=float, default=0.1)
    parser.add_argument('--n_pos', type=int, default=9)
    parser.add_argument('--n_neg', type=int, default=30)

    # ---- SimCLR ----
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--simclr_weight', type=float, default=1.0)
    parser.add_argument('--n_views', default=2, type=int)

    # ---- misc ----
    parser.add_argument('-n', '--model_name', type=str, default='SixStudy_supconH_SimCLR_extraFeat')
    parser.add_argument('--adaptive_rate', type=int, default=6)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--include_zero_gene', action='store_true')
    parser.add_argument('--max_length_esm2', default=25, type=int)
    parser.add_argument('--length_range', default=25, type=int)
    parser.add_argument('--load_local_pretrain', type=str, default=None)
    parser.add_argument('--load_local_esm2_pretrain', type=str, default=None)
    parser.add_argument('--no_timestamp', action='store_true')
    parser.add_argument('--save_eval_interval', default=10, type=int)
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

train_adata = sc.read_h5ad(args.train_data_path)
val_adata   = sc.read_h5ad(args.val_data_path)
test_adata  = sc.read_h5ad(args.test_data_path)

train_adata.var["gene_name"] = train_adata.var.index.tolist()
val_adata.var["gene_name"]   = val_adata.var.index.tolist()
num_types = 2

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

hyperparameter_defaults = dict(
    seed=0,
    model_name=args.model_name,
    max_seq_len=args.max_seq_len,
    temp=args.temp,
    temperature_simclr=args.temperature,
    n_pos=args.n_pos,
    n_neg=args.n_neg,
    hidden_dim=args.hidden_dim,
    out_dim=args.out_dim,
    adaptive_rate=args.adaptive_rate,
    verbose=args.verbose,
    do_train=True,
    load_model=scgpt_model_path,
    load_local_pretrain=args.load_local_pretrain,
    load_local_esm2_pretrain=args.load_local_esm2_pretrain,
    unfix_last_layer=args.unfix_last_layer,
    mask_ratio=0.0,
    epochs=args.epochs,
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    dab_weight=0.0,
    lr=args.lr,
    lr_seq=args.lr_seq,
    optimi=args.optimi,
    first_cycle_steps=args.first_cycle_steps,
    warmup=args.warmup,
    warmup_gamma=args.warmup_gamma,
    batch_size=args.batch_size,
    layer_size=128,
    nlayers=4,
    nlayers_cls=args.nlayers_cls,
    nhead=4,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=args.save_eval_interval,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=args.include_zero_gene,
    freeze=True,
    DSBN=False,
    extra_feat_dim=args.extra_feat_dim,
    simclr_weight=args.simclr_weight,
)
run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)

set_seed(config.seed)

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio  = config.mask_ratio
mask_value  = "auto"
include_zero_gene = config.include_zero_gene
max_seq_len = config.max_seq_len
n_bins      = config.n_bins

input_style   = "log1p"
output_style  = "binned"
MLM = False; CLS = True; ADV = False; CCE = False
MVC = config.MVC
ECS = config.ecs_thres > 0
DAB = False; INPUT_BATCH_LABELS = False
input_emb_style  = "continuous"
cell_emb_style   = "cls"
mvc_decoder_style = "inner product"
ecs_threshold    = config.ecs_thres
dab_weight       = config.dab_weight
num_batch_types  = None
explicit_zero_prob   = MLM and include_zero_gene
do_sample_in_train   = False and explicit_zero_prob
per_seq_batch_sample = False

lr          = config.lr
batch_size  = config.batch_size
epochs      = config.epochs
fast_transformer         = config.fast_transformer
fast_transformer_backend = "flash"
embsize  = config.layer_size
d_hid    = config.layer_size
nlayers  = config.nlayers
nhead    = config.nhead
dropout  = config.dropout
log_interval      = 100
save_eval_interval = config.save_eval_interval

if input_emb_style == "category":
    mask_value  = n_bins + 1
    pad_value   = n_bins
    n_input_bins = n_bins + 2
else:
    mask_value  = -1
    pad_value   = -2
    n_input_bins = n_bins

DAB_separate_optim = True if DAB > 1 else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32

model_name = config.model_name
gene_panel_length = Path(args.train_data_path).parent.name

if not args.no_timestamp:
    save_dir = Path(
        f"{os.getcwd()}/save/dev_{model_name}_glen{gene_panel_length}"
        f"_unfix{config.unfix_last_layer}_lr{config.lr}_lrseq{config.lr_seq}"
        f"_npos{config.n_pos}_nneg{config.n_neg}_t{config.temp}_maxl{max_seq_len}"
        f"_simclr_wght{args.simclr_weight}_extrafeat{args.extra_feat_dim}"
        f"-{time.strftime('%b%d-%H-%M')}/"
    )
else:
    save_dir = Path(
        f"{os.getcwd()}/save/dev_{model_name}_glen{gene_panel_length}"
        f"_unfix{config.unfix_last_layer}_lr{config.lr}_lrseq{config.lr_seq}"
        f"_npos{config.n_pos}_nneg{config.n_neg}_t{config.temp}_maxl{max_seq_len}"
        f"_simclr_wght{args.simclr_weight}_extrafeat{args.extra_feat_dim}-SuContinue/"
    )
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

data_is_raw   = False
model_config_file = Path(scgpt_model_path) / "args.json"
vocab_file        = Path(scgpt_model_path) / "vocab.json"

vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

train_adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in train_adata.var["gene_name"]]
gene_ids_in_vocab = np.array(train_adata.var["id_in_vocab"])
logger.info(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocab of size {len(vocab)}.")
train_adata = train_adata[:, train_adata.var["id_in_vocab"] >= 0]

val_adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in val_adata.var["gene_name"]]
val_adata = val_adata[:, val_adata.var["id_in_vocab"] >= 0]

with open(model_config_file, "r") as f:
    model_configs = json.load(f)
embsize  = model_configs["embsize"]
nhead    = model_configs["nheads"]
d_hid    = model_configs["d_hid"]
nlayers  = model_configs["nlayers"]

preprocessor = Preprocessor(
    use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
    normalize_total=False, result_normed_key="X", log1p=data_is_raw,
    result_log1p_key="X", subset_hvg=False,
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins, result_binned_key="X_binned",
)
preprocessor(train_adata, batch_key=None)
preprocessor(test_adata, batch_key=None)
preprocessor(val_adata, batch_key=None)

train_adata_pos = train_adata[train_adata.obs['reactivity'] == '1']
train_adata_neg = train_adata[train_adata.obs['reactivity'] == '0']
val_adata_pos   = val_adata[val_adata.obs['reactivity'] == '1']
val_adata_neg   = val_adata[val_adata.obs['reactivity'] == '0']

input_layer_key = "X_binned"
config.input_layer_key = input_layer_key


def _return_count_data(adata):
    x = adata.layers[input_layer_key]
    return x.A if issparse(x) else x


train_pos_data = _return_count_data(train_adata_pos)
train_neg_data = _return_count_data(train_adata_neg)
val_pos_data   = _return_count_data(val_adata_pos)
val_neg_data   = _return_count_data(val_adata_neg)

genes = train_adata.var["gene_name"].tolist()
if config.load_model is None:
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

ntokens = len(vocab)
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

if config.load_local_pretrain is not None:
    model.cls_decoder = LayerNormNet(config.hidden_dim, config.out_dim)
    model_file = Path(config.load_local_pretrain)
    ckpt = torch.load(model_file, map_location='cpu')
    pretrained_state_dic = ckpt.get('state_dict1', ckpt)
    try:
        model.load_state_dict(pretrained_state_dic)
        logger.info(f"Loading all model params from {model_file}")
    except:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_state_dic.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
else:
    model_file = Path(config.load_model) / "best_model.pt"
    try:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        logger.info(f"Loading all model params from {model_file}")
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location='cpu')
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.cls_decoder = LayerNormNet(config.hidden_dim, config.out_dim)

pre_freeze_param_count = sum(
    dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()
)
for name, para in model.named_parameters():
    if config.freeze and "encoder" in name:
        para.requires_grad = False
fix_layer_num = nlayers - config.unfix_last_layer
fix_layer_index = 0
for layer in model.transformer_encoder.layers:
    if fix_layer_index < fix_layer_num:
        fix_layer_index += 1
        continue
    for p in layer.parameters():
        p.requires_grad = True

post_freeze_param_count = sum(
    dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()
)
logger.info(f"Total Pre freeze Params {pre_freeze_param_count}")
logger.info(f"Total Post freeze Params {post_freeze_param_count}")
model.to(device)
wandb.watch(model)

ESM2_OUT_DIM = 1280
model_seq = MoBYMLP(
    in_dim=ESM2_OUT_DIM + config.extra_feat_dim,
    inner_dim=2048,
    num_layers=2,
    out_dim=config.out_dim,
)

if config.load_local_esm2_pretrain is not None:
    model_file_esm2 = Path(config.load_local_esm2_pretrain)
    ckpt2 = torch.load(model_file_esm2, map_location='cpu')
    pretrained_state_dic = ckpt2.get('state_dict2', ckpt2)
    try:
        model_seq.load_state_dict(pretrained_state_dic)
        logger.info(f"Loading model_seq params from {model_file_esm2}")
    except:
        model_dict = model_seq.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_state_dic.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model_seq.load_state_dict(model_dict)

for name, para in model_seq.named_parameters():
    para.requires_grad = True
model_seq_param_count = sum(
    dict((p.data_ptr(), p.numel()) for p in model_seq.parameters() if p.requires_grad).values()
)
print(f"Total model_seq Params {model_seq_param_count}")
model_seq.to(device)
wandb.watch(model_seq)


# ============================================================ #
# Data tokenisation
# ============================================================ #
train_pos_tokenized = tokenize_and_pad_batch(
    train_pos_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=pad_token, pad_value=pad_value, append_cls=True,
    include_zero_gene=include_zero_gene,
)
train_neg_tokenized = tokenize_and_pad_batch(
    train_neg_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=pad_token, pad_value=pad_value, append_cls=True,
    include_zero_gene=include_zero_gene,
)
val_pos_tokenized = tokenize_and_pad_batch(
    val_pos_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=pad_token, pad_value=pad_value, append_cls=True,
    include_zero_gene=include_zero_gene,
)
val_neg_tokenized = tokenize_and_pad_batch(
    val_neg_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=pad_token, pad_value=pad_value, append_cls=True,
    include_zero_gene=include_zero_gene,
)


def prepare_data(pos_tok, neg_tok, adata_pos, adata_neg):
    masked_pos = random_mask_value(pos_tok["values"], mask_ratio=mask_ratio,
                                   mask_value=mask_value, pad_value=pad_value)
    masked_neg = random_mask_value(neg_tok["values"], mask_ratio=mask_ratio,
                                   mask_value=mask_value, pad_value=pad_value)
    pos_data_pt = {
        "gene_ids":      pos_tok["genes"],
        "values":        masked_pos,
        "target_values": pos_tok["values"],
        "cell_ids":      list(adata_pos.obs_names),
    }
    neg_data_pt = {
        "gene_ids":      neg_tok["genes"],
        "values":        masked_neg,
        "target_values": neg_tok["values"],
        "cell_ids":      list(adata_neg.obs_names),
    }
    return pos_data_pt, neg_data_pt


class CustomDataset(Dataset):
    def __init__(self, positive_data, negative_data, n_pos, n_neg):
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.positive_samples = self.create_samples()

    def create_samples(self):
        positive_samples = []
        for idx, anchor_id in enumerate(self.positive_data["cell_ids"]):
            sample = [{"gene_ids": self.positive_data["gene_ids"][idx],
                       "values":   self.positive_data["values"][idx]}]
            pos_indices = [i for i, cid in enumerate(self.positive_data["cell_ids"])
                           if cid != anchor_id]
            pos_samples = random.sample(pos_indices, min(self.n_pos, len(pos_indices)))
            sample.extend([{"gene_ids": self.positive_data["gene_ids"][i],
                             "values":  self.positive_data["values"][i]}
                           for i in pos_samples])
            neg_indices = list(range(len(self.negative_data["cell_ids"])))
            neg_samples = random.sample(neg_indices, min(self.n_neg, len(neg_indices)))
            sample.extend([{"gene_ids": self.negative_data["gene_ids"][i],
                             "values":  self.negative_data["values"][i]}
                           for i in neg_samples])
            positive_samples.append(sample)
        return positive_samples

    def __len__(self):
        return len(self.positive_samples)

    def __getitem__(self, idx):
        return self.positive_samples[idx]


def collate_fn(batch):
    batched_data = {'gene_ids': [], 'values': []}
    batch_size = len(batch)
    num_dicts  = len(batch[0])
    for data_point in batch:
        for d in data_point:
            batched_data['gene_ids'].append(d['gene_ids'])
            batched_data['values'].append(d['values'])
    batched_data['gene_ids'] = torch.stack(batched_data['gene_ids']).view(batch_size, num_dicts, -1)
    batched_data['values']   = torch.stack(batched_data['values']).view(batch_size, num_dicts, -1)
    return batched_data


# ============================================================ #
# Training loop
# ============================================================ #
def train(mdl, params, epc, train_data_loader,
          opt, dev, dt, loss_fn, mdl_seq, train_loader_simclr_, opt_seq, cfg):
    mdl.train()
    mdl_seq.train()
    total_loss = 0.
    start_time = time.time()

    loader1_iter = iter(train_data_loader)
    loader2_iter = iter(train_loader_simclr_)
    max_len = len(train_data_loader)

    for batch_index_ in range(max_len):
        data1 = next(loader1_iter, None)
        if data1 is None:
            loader1_iter = iter(train_data_loader)
            data1 = next(loader1_iter)

        data2 = next(loader2_iter, None)
        if data2 is None:
            loader2_iter = iter(train_loader_simclr_)
            data2 = next(loader2_iter)

        opt.zero_grad()
        opt_seq.zero_grad()
        embeddings = torch.zeros((params.batch_size, 1 + params.n_pos + params.n_neg, params.out_dim))

        for i in range(params.batch_size):
            gene_ids_ = data1['gene_ids'][i].to(dev)
            values_   = data1['values'][i].to(dev)
            src_key_padding_mask_ = gene_ids_.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                output_dict = mdl(
                    gene_ids_, values_,
                    src_key_padding_mask=src_key_padding_mask_,
                    batch_labels=None,
                    CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS,
                    do_sample=do_sample_in_train,
                )
            embeddings[i] = output_dict["cls_output"]

        loss_gex, a_pos_sim, a_neg_sim = loss_fn(embeddings, params.temp, params.n_pos, params.n_neg)

        simclr = SimCLRWithExtraFeat(
            extra_feat_dim=cfg.extra_feat_dim,
            model_gex=mdl, model_seq=mdl_seq,
            optimizer_gex=opt, optimizer_seq=opt_seq,
            args=params, device=dev,
            vocab=vocab, pad_token=pad_token, pad_value=pad_value,
            config=cfg,
        )

        batch = data2[0]
        bsz = len(batch)
        seq_input_tensor = build_seq_input(batch, cfg.extra_feat_dim, dev)

        gex_input = [(d['genes_data'], d['expressions_data']) for d in batch]
        batch_genes_data, batch_expressions_data = zip(*gex_input)
        bg_tensor = torch.full((bsz, max_seq_len), vocab[pad_token], dtype=torch.int64)
        be_tensor = torch.full((bsz, max_seq_len), pad_value, dtype=torch.float32)
        for i, (gd, ed) in enumerate(zip(batch_genes_data, batch_expressions_data)):
            bg_tensor[i, :len(gd)] = gd
            be_tensor[i, :len(ed)] = ed
        bg_tensor = bg_tensor.to(dev)
        be_tensor = be_tensor.to(dev)

        with autocast(enabled=cfg.amp):
            src_key_padding_mask = bg_tensor.eq(vocab[pad_token])
            output_dict = mdl(
                bg_tensor, be_tensor,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS,
                do_sample=do_sample_in_train,
            )
            features_gex = output_dict["cls_output"]
            features_seq = mdl_seq(seq_input_tensor)
            logits, labels = simclr.info_nce_loss(features_gex, features_seq)
            simclr_loss = simclr.criterion(logits, labels)

        total_loss = loss_gex + params.simclr_weight * simclr_loss
        total_loss.backward()
        opt.step()
        opt_seq.step()

        l_prob        = logits[:, 0].mean()
        negsim_gex_gex = log_negative_mean_logtis(logits, "struct_struct", bsz)
        negsim_gex_seq = log_negative_mean_logtis(logits, "struct_seq", bsz)
        negsim_seq_seq = log_negative_mean_logtis(logits, "seq_seq", bsz)

        wandb.log({
            "train/supconloss":      loss_gex.item(),
            "train/anchor_pos_sim":  a_pos_sim.item(),
            "train/anchor_neg_sim":  a_neg_sim.item(),
            "train/simclr_loss":     simclr_loss.item(),
            "train/total_loss":      total_loss.item(),
            "train/l_prob":          l_prob.item(),
            "train/negsim_gex_gex":  negsim_gex_gex,
            "train/negsim_gex_seq":  negsim_gex_seq,
            "train/negsim_seq_seq":  negsim_seq_seq,
        })

        if params.verbose:
            ms_per_batch = (time.time() - start_time) * 1000
            print(f'| epoch {epc:3d} | {batch_index_:5d}/{len(train_data_loader):5d} batches | '
                  f'lr {params.lr:02.4f} | ms/batch {ms_per_batch:6.4f} | loss {total_loss:5.2f}')
            start_time = time.time()

    return total_loss / (batch_index_ + 1)


# ============================================================ #
# Evaluation
# ============================================================ #
def define_wandb_metrcis():
    wandb.define_metric("epoch")
    for prefix in ["validation", "testing"]:
        for m in ["knn_7_f1", "knn_7_auc", "knn_7_ari", "nc_f1", "nc_auc", "nc_ari", "ch_index"]:
            wandb.define_metric(f"{prefix}/{m}", summary="max", step_metric="epoch")
    wandb.define_metric("validation/supconloss", summary="min", step_metric="epoch")
    wandb.define_metric("validation/anchor_pos_sim", summary="max", step_metric="epoch")
    wandb.define_metric("validation/anchor_neg_sim", summary="min", step_metric="epoch")
    wandb.define_metric("validation/center_distance", summary="max", step_metric="epoch")


def save_checkpoint(state, filename=None):
    if filename is None:
        filename = save_dir / 'latest_ck.pt'
    torch.save(state, filename)


def load_checkpoint(mdl, optim, mdl_seq, optim_seq, filename=None):
    if filename is None:
        filename = save_dir / 'latest_ck.pt'
    ckpt = torch.load(filename, map_location='cpu' if not torch.cuda.is_available() else None)
    mdl.load_state_dict(ckpt['state_dict1'])
    optim.load_state_dict(ckpt['optimizer'])
    mdl_seq.load_state_dict(ckpt['state_dict2'])
    optim_seq.load_state_dict(ckpt['optimizer_seq'])
    start_e = ckpt['epoch'] + 1
    best_loss = ckpt.get('best_loss', float('inf'))
    if 'best_mdl_state_dict' in ckpt:
        best_model = copy.deepcopy(mdl)
        best_model.load_state_dict(ckpt['best_mdl_state_dict'])
    else:
        best_model = None
    best_model_epoch = ckpt.get('best_model_epoch', None)
    return mdl, optim, mdl_seq, optim_seq, start_e, best_loss, best_model, best_model_epoch


def evaluate(cfg, mdl, params, epc, val_data, val_data_loader, train_data, dev, dt,
             loss_fn, actual_val_data, out_prefix, ret_loss, mdl_seq, val_loader_simclr_,
             opt, opt_seq):
    mdl.eval()
    mdl_seq.eval()
    total_loss = total_a_pos_sim = total_a_neg_sim = 0.
    metrics_to_log = {}

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.amp):
        val_data_   = get_project_emb(cfg, mdl, val_data,   gene_ids, max_seq_len, vocab, pad_token, pad_value, include_zero_gene, dev)
        train_data_ = get_project_emb(cfg, mdl, train_data, gene_ids, max_seq_len, vocab, pad_token, pad_value, include_zero_gene, dev)
        actual_val_data_ = (
            get_project_emb(cfg, mdl, actual_val_data, gene_ids, max_seq_len, vocab, pad_token, pad_value, include_zero_gene, dev)
            if out_prefix == 'testing' else val_data_.copy()
        )

        unique_labels = list(train_data_.obs['reactivity'].unique())
        predictions, function_names = [], []

        for pref_, obsm_key in zip(["projector"], ["X_scGPT_prj"]):
            class_centers          = get_cluster_center(train_data_, obsm_key, False)
            class_centers_norm_val = get_cluster_center(actual_val_data_, obsm_key, True)

            train_embeddings_np = train_data_.obsm[obsm_key]
            val_embeddings_np   = val_data_.obsm[obsm_key]
            actual_val_embeddings_np = actual_val_data_.obsm[obsm_key]

            train_labels_np = train_data_.obs["reactivity"].values
            val_labels_np   = val_data_.obs["reactivity"].values
            actual_val_labels_np = actual_val_data_.obs["reactivity"].values

            pro_class_centers_distance_val = np.linalg.norm(
                class_centers_norm_val['0'] - class_centers_norm_val['1']
            )
            pro_ch_index = calinski_harabasz_score(val_embeddings_np, val_labels_np)

            for n in range(1, 11):
                predictions.append(knn_classifier(n, train_embeddings_np, train_labels_np, val_embeddings_np))

            predictions.append(nearest_center(val_embeddings_np, class_centers))
            predictions.append(cosine_similarity_classifier(train_embeddings_np, val_embeddings_np, train_labels_np, unique_labels))

            (pro_Youdensj_threshold, pro_Youdensj_predictions,
             pro_jenks_threshold, pro_jenks_predictions,
             pro_distance_prediction_score) = distance_based_prediction(
                val_embeddings_np, class_centers, val_labels_np,
                actual_val_embeddings_np, actual_val_labels_np
            )
            metrics_to_log.update({f"{out_prefix}/{pref_}_Youdensj_threshold": pro_Youdensj_threshold})
            metrics_to_log.update({f"{out_prefix}/{pref_}_jenks_threshold":    pro_jenks_threshold})

            predictions.append((pro_Youdensj_predictions, pro_distance_prediction_score))
            predictions.append((pro_jenks_predictions, pro_distance_prediction_score))
            predictions.append((pro_jenks_predictions, pro_distance_prediction_score))

            function_names += (
                [pref_ + '_knn_' + str(n) for n in range(1, 11)] +
                [pref_ + "_nearest_center", pref_ + "_cosine_similarity",
                 pref_ + "_distance_Youdensj", pref_ + "_distance_jenks", pref_ + "_svm"]
            )

            OT_prediction_list, OT_function_list = OT_based_prediction(
                train_embeddings_np, val_embeddings_np, train_labels_np
            )
            predictions    += OT_prediction_list
            function_names += OT_function_list

        assert len(predictions) == len(function_names)

        val_data_.obs['pro_knn_7']                    = predictions[6][0]
        val_data_.obs['pro_nearest_center_preds']      = predictions[10][0]
        val_data_.obs['pro_cosine_sim']                = predictions[11][0]
        val_data_.obs['pro_dYoudensj']                 = predictions[12][0]
        val_data_.obs['pro_dJenks']                    = predictions[13][0]
        val_data_.obs['pro_svm']                       = predictions[14][0]
        val_data_.obs['pro_top10_OT_KNN_Weights_30best']   = predictions[15][0]
        val_data_.obs['pro_jenks_OT_KNN_Weights_30best']   = predictions[16][0]
        val_data_.obs['pro_top10_OT_deltarho_30best']      = predictions[17][0]
        val_data_.obs['pro_jenks_OT_deltarho_30best']      = predictions[18][0]

        output_file = save_dir / f"{out_prefix}_prediction_scores_e{epc}.csv"
        results_list = output_metrics(output_file, predictions, function_names, val_labels_np)

        def _r(idx): return results_list[idx]
        pro_knn_7_f1 = _r(6)[2];  pro_knn_7_auc = _r(6)[3];  pro_knn_7_ari = _r(6)[6]
        pro_knn_7_precision = _r(6)[0]; pro_knn_7_recall = _r(6)[1]
        pro_knn_7_accuracy  = _r(6)[4]; pro_knn_7_mcc    = _r(6)[7]; pro_knn_7_gmean = _r(6)[8]
        pro_nc_f1  = _r(10)[2]; pro_nc_auc  = _r(10)[3]; pro_nc_ari  = _r(10)[6]
        pro_nc_precision = _r(10)[0]; pro_nc_recall = _r(10)[1]
        pro_nc_accuracy  = _r(10)[4]; pro_nc_mcc    = _r(10)[7]; pro_nc_gmean = _r(10)[8]
        pro_cosine_f1 = _r(11)[2]; pro_cosine_auc = _r(11)[3]
        pro_dYoudensj_f1 = _r(12)[2]; pro_dYoudensj_auc = _r(12)[3]
        pro_dJenks_f1 = _r(13)[2]; pro_dJenks_auc = _r(13)[3]
        pro_svm_f1 = _r(14)[2]; pro_svm_auc = _r(14)[3]

        for key, val in [
            (f"{out_prefix}/projector_center_distance", pro_class_centers_distance_val),
            (f"{out_prefix}/pro_ch_index",       pro_ch_index),
            (f"{out_prefix}/pro_knn_7_f1",        pro_knn_7_f1),
            (f"{out_prefix}/pro_knn_7_auc",       pro_knn_7_auc),
            (f"{out_prefix}/pro_knn_7_ari",       pro_knn_7_ari),
            (f"{out_prefix}/pro_knn_7_mcc",       pro_knn_7_mcc),
            (f"{out_prefix}/pro_knn_7_precision", pro_knn_7_precision),
            (f"{out_prefix}/pro_knn_7_recall",    pro_knn_7_recall),
            (f"{out_prefix}/pro_knn_7_accuracy",  pro_knn_7_accuracy),
            (f"{out_prefix}/pro_knn_7_gmean",     pro_knn_7_gmean),
            (f"{out_prefix}/pro_nc_f1",           pro_nc_f1),
            (f"{out_prefix}/pro_nc_auc",          pro_nc_auc),
            (f"{out_prefix}/pro_nc_ari",          pro_nc_ari),
            (f"{out_prefix}/pro_nc_mcc",          pro_nc_mcc),
            (f"{out_prefix}/pro_cosine_f1",       pro_cosine_f1),
            (f"{out_prefix}/pro_cosine_auc",      pro_cosine_auc),
            (f"{out_prefix}/pro_dYoudensj_f1",    pro_dYoudensj_f1),
            (f"{out_prefix}/pro_dJenks_f1",       pro_dJenks_f1),
            (f"{out_prefix}/pro_svm_f1",          pro_svm_f1),
        ]:
            metrics_to_log[key] = val

        functions_ = ['top10_OT_KNN_Weights_30best', 'jenks_OT_KNN_Weights_30best',
                      'top10_OT_deltarho_30best', 'jenks_OT_deltarho_30best']
        metrics_names_ = ["precision", "recall", "f1", "auc", "accuracy", "Balanced_accuracy", "ari", "mcc", "gmean"]
        for func_name, idx in zip(functions_, [15, 16, 17, 18]):
            for m_idx, m_name in enumerate(metrics_names_):
                if m_idx == 5:
                    continue
                metrics_to_log[f"{out_prefix}/pro_{func_name}_{m_name}"] = results_list[idx][m_idx]

        for fig_type, pl_fn in [("umap", sc.pl.umap), ("tsne", sc.pl.tsne)]:
            fig_path = save_dir / f"{out_prefix}_proj_embeddings_reactivity_{fig_type}_e{epc}.png"
            with plt.rc_context({"figure.figsize": (10, 2), "figure.dpi": (100)}):
                pl_fn(
                    val_data_,
                    color=["reactivity", 'pro_knn_7', 'pro_nearest_center_preds', 'pro_cosine_sim',
                           'pro_dYoudensj', 'pro_dJenks', 'pro_svm',
                           "pro_top10_OT_KNN_Weights_30best", "pro_jenks_OT_KNN_Weights_30best",
                           "pro_top10_OT_deltarho_30best", "pro_jenks_OT_deltarho_30best"],
                    title=[f'projector CH: {pro_ch_index:.2f}'],
                    show=False,
                )
                plt.savefig(fig_path, dpi=100)

        sc.pp.neighbors(val_data_, use_rep="X_scGPT_trained")
        sc.tl.umap(val_data_, min_dist=0.3)
        sc.tl.tsne(val_data_, use_rep="X_scGPT_trained")

        if out_prefix == 'testing':
            for fig_type in ["umap", "tsne"]:
                metrics_to_log[f"{out_prefix}/scGPT_prj_cell_{fig_type}"] = wandb.Image(
                    str(save_dir / f"{out_prefix}_proj_embeddings_reactivity_{fig_type}_e{epc}.png"),
                    caption=f"{out_prefix} pro_ch_index:{pro_ch_index:.2f}, pro_center_dist:{pro_class_centers_distance_val:.3f}",
                )

        if ret_loss:
            for batch, data in enumerate(val_data_loader):
                embeddings = torch.zeros((params.batch_size, 1 + params.n_pos + params.n_neg, params.out_dim))
                for i in range(params.batch_size):
                    gid_ = data['gene_ids'][i].to(dev)
                    val_ = data['values'][i].to(dev)
                    mask_ = gid_.eq(vocab[pad_token])
                    with torch.cuda.amp.autocast(enabled=cfg.amp):
                        out = mdl(gid_, val_, src_key_padding_mask=mask_,
                                  batch_labels=None,
                                  CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS, do_sample=do_sample_in_train)
                        embeddings[i] = out["cls_output"]
                loss, a_pos_sim, a_neg_sim = loss_fn(embeddings, params.temp, params.n_pos, params.n_neg)
                total_loss += loss.item()
                total_a_pos_sim += a_pos_sim.item()
                total_a_neg_sim += a_neg_sim.item()

            averaged_loss      = total_loss / (batch + 1)
            averaged_a_pos_sim = total_a_pos_sim / (batch + 1)
            averaged_a_neg_sim = total_a_neg_sim / (batch + 1)
            metrics_to_log.update({
                f"{out_prefix}/supconloss":     averaged_loss,
                f"{out_prefix}/anchor_pos_sim": averaged_a_pos_sim,
                f"{out_prefix}/anchor_neg_sim": averaged_a_neg_sim,
            })

            simclr_eval = SimCLRWithExtraFeat(
                extra_feat_dim=cfg.extra_feat_dim,
                model_gex=mdl, model_seq=mdl_seq,
                optimizer_gex=opt, optimizer_seq=opt_seq,
                args=params, device=dev,
                vocab=vocab, pad_token=pad_token, pad_value=pad_value,
                config=cfg,
            )
            loss_val, val_l_prob, val_ng_gg, val_ng_gs, val_ng_ss = simclr_eval.validation(val_loader_simclr_)
            val_total_loss = averaged_loss + params.simclr_weight * loss_val
            metrics_to_log.update({
                f"{out_prefix}/simclr_loss":    loss_val,
                f"{out_prefix}/total_loss":     val_total_loss,
                f"{out_prefix}/l_prob":         val_l_prob,
                f"{out_prefix}/negsim_gex_gex": val_ng_gg,
                f"{out_prefix}/negsim_gex_seq": val_ng_gs,
                f"{out_prefix}/negsim_seq_seq": val_ng_ss,
            })

            logger.info(
                f"[{out_prefix}] epoch {epc} | supcon {averaged_loss:.4f} | "
                f"simclr {loss_val:.4f} | total {val_total_loss:.4f} | "
                f"knn7_f1 {pro_knn_7_f1:.4f} | knn7_auc {pro_knn_7_auc:.4f}"
            )

            if epc % cfg.save_eval_interval == 0 or epc == cfg.epochs:
                val_data_.write_h5ad(save_dir / f"{out_prefix}_val_predictions_e{epc}.h5ad")

        wandb.log(metrics_to_log)

    if ret_loss:
        return averaged_loss, pro_class_centers_distance_val
    else:
        return pro_class_centers_distance_val


# ============================================================ #
# Main
# ============================================================ #
def main(model, model_seq):
    torch.backends.cudnn.benchmark = True
    lr, epochs = args.lr, args.epochs
    print('==> device:', device, '| dtype:', dtype, '\n==> args:', args)

    optimizer     = torch.optim.Adam(model.parameters(),     lr=lr,         eps=1e-4 if config.amp else 1e-8)
    optimizer_seq = torch.optim.Adam(model_seq.parameters(), lr=args.lr_seq, eps=1e-4 if config.amp else 1e-8)

    if config.load_local_pretrain is not None:
        ckpt = torch.load(Path(config.load_local_pretrain),
                          map_location='cpu' if not torch.cuda.is_available() else None)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'optimizer_seq' in ckpt:
            optimizer_seq.load_state_dict(ckpt['optimizer_seq'])

    criterion = SupConHardLoss
    best_loss = float('inf')
    best_model = None
    define_wandb_metrcis()

    val_pos_data_pt, val_neg_data_pt = prepare_data(val_pos_tokenized, val_neg_tokenized, val_adata_pos, val_adata_neg)
    val_dataset = CustomDataset(val_pos_data_pt, val_neg_data_pt, args.n_pos, args.n_neg)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, drop_last=True)

    print("Loading tar files...")
    t1 = time.time()
    train_dataset_simclr = wds.WebDataset(args.train_data_path_simclr).shuffle(2000).decode().to_tuple("npz")
    val_dataset_simclr   = wds.WebDataset(args.val_data_path_simclr).decode().to_tuple("npz")
    train_dataset_simclr = train_dataset_simclr.batched(args.batch_size)
    val_dataset_simclr   = val_dataset_simclr.batched(args.batch_size)
    print(f"Tar files loaded in {time.time()-t1:.1f}s")

    train_loader_simclr = wds.WebLoader(train_dataset_simclr, batch_size=None, shuffle=False, pin_memory=False)
    val_loader_simclr   = wds.WebLoader(val_dataset_simclr,   batch_size=None, shuffle=False, pin_memory=False)
    val_loader_simclr   = val_loader_simclr.unbatched().batched(args.batch_size, partial=False)

    start_epoch = 1
    checkpoint_file = save_dir / 'latest_ck.pt'
    if os.path.exists(checkpoint_file):
        print("Resuming from checkpoint...")
        model, optimizer, model_seq, optimizer_seq, start_epoch, best_loss, best_model, best_model_epoch = \
            load_checkpoint(model, optimizer, model_seq, optimizer_seq, checkpoint_file)
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        pos_data_pt, neg_data_pt = prepare_data(train_pos_tokenized, train_neg_tokenized, train_adata_pos, train_adata_neg)
        dataset      = CustomDataset(pos_data_pt, neg_data_pt, args.n_pos, args.n_neg)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=True)
        train_loader_simclr = train_loader_simclr.unbatched().shuffle(1000).batched(args.batch_size, partial=False)

        t_epoch = time.time()
        train_loss = train(model, args, epoch, train_loader, optimizer, device, dtype,
                           criterion, model_seq, train_loader_simclr, optimizer_seq, config)
        t_train = time.time() - t_epoch

        val_loss, val_pro_centers_distance = evaluate(
            config, model, args, epoch, val_adata, val_loader, train_adata,
            device, dtype, criterion, val_adata, "validation", True,
            model_seq, val_loader_simclr, optimizer, optimizer_seq
        )
        elapsed = time.time() - t_epoch

        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed/60:.2f}m | train: {t_train/60:.2f}m | "
            f"valid loss {val_loss:.4f} | centers dist {val_pro_centers_distance:.2f}"
        )
        logger.info("-" * 89)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best from epoch : {epoch:3d}; loss: {best_loss:6.4f}")

        ckpt_state = {
            'epoch': epoch,
            'state_dict1': model.state_dict(),
            'state_dict2': model_seq.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'optimizer_seq': optimizer_seq.state_dict(),
            'best_loss':     best_loss,
            'best_mdl_state_dict': best_model.state_dict(),
            'best_model_epoch':    best_model_epoch,
            'extra_feat_dim': config.extra_feat_dim,
        }
        save_checkpoint(ckpt_state)

        if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
            save_checkpoint(ckpt_state, filename=save_dir / f'model_e{epoch}.pt')
            evaluate(config, model, args, epoch, test_adata, val_loader, train_adata,
                     device, dtype, criterion, val_adata, "testing", False,
                     model_seq, val_loader_simclr, optimizer, optimizer_seq)

    evaluate(config, best_model, args, best_model_epoch, test_adata, val_loader, train_adata,
             device, dtype, criterion, val_adata, "testing", False,
             model_seq, val_loader_simclr, optimizer, optimizer_seq)
    logger.info(f"Best epoch: {best_model_epoch}")

    run.finish()
    wandb.finish()
    gc.collect()


if __name__ == '__main__':
    main(model, model_seq)
