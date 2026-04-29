# scripts/train_supcon.py
#
# SupCon-Hard loss training using scGPT gene-expression embeddings only.
# No SimCLR cross-modal branch; no webdataset / ESM2 tar files needed.
#
# EXAMPLE COMMAND:
#   python scripts/train_supcon.py \
#       --train_data_path /path/to/train.h5ad \
#       --val_data_path   /path/to/val.h5ad \
#       --test_data_path  /path/to/test.h5ad \
#       --lr 1e-4  --epochs 100  --batch_size 32 \
#       --n_pos 9  --n_neg 30   --temp 0.1 \
#       --unfix_last_layer 1  --out_dim 128  --hidden_dim 256 \
#       --model_name MyExperiment_SupCon
#
# TUNABLE PARAMETERS:
#   --lr               learning rate (default 1e-4)
#   --epochs           number of epochs (default 100)
#   --batch_size       batch size (default 32)
#   --temp             SupCon-Hard temperature (default 0.1)
#   --n_pos            positives per anchor (default 9)
#   --n_neg            negatives per anchor (default 30)
#   --unfix_last_layer unfreeze last N scGPT transformer layers (default 0)
#   --out_dim          projector output dim (default 128)
#   --hidden_dim       cls_decoder hidden dim (default 256)
#   --save_eval_interval checkpoint/eval every N epochs (default 10)

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
from torchtext._torchtext import Vocab as VocabPybind

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
from scgpt.loss import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

# ---- scTRP package imports ----
from scTRP.inference.functions import (
    return_count_data, get_project_emb, get_cluster_center,
    knn_classifier, nearest_center, cosine_similarity_classifier,
    distance_based_prediction, OT_based_prediction, output_metrics,
)
from scTRP.models.layers import LayerNormNet
from scTRP.losses import SupConHardLoss

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lr', type=float, default=1e-4)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-n', '--model_name', type=str, default='SixStudy_supconH')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_len', default=1200, type=int)
    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('-T', '--temp', type=float, default=0.1)
    parser.add_argument('--n_pos', type=int, default=9)
    parser.add_argument('--n_neg', type=int, default=30)
    parser.add_argument('-d', '--hidden_dim', type=int, default=256)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=6)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--include_zero_gene', action='store_true')
    parser.add_argument('--optimi', type=str, default="Adam")
    parser.add_argument('--load_local_pretrain', type=str, default=None)
    parser.add_argument('--warmup', default=50, type=int)
    parser.add_argument('--first_cycle_steps', default=3162, type=int)
    parser.add_argument('--warmup_gamma', default=1.0, type=float)
    parser.add_argument('--unfix_last_layer', type=int, default=0)
    parser.add_argument('--nlayers_cls', type=int, default=3)
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
    n_pos=args.n_pos,
    n_neg=args.n_neg,
    hidden_dim=args.hidden_dim,
    out_dim=args.out_dim,
    adaptive_rate=args.adaptive_rate,
    verbose=args.verbose,
    do_train=True,
    load_model=scgpt_model_path,
    load_local_pretrain=args.load_local_pretrain,
    unfix_last_layer=args.unfix_last_layer,
    mask_ratio=0.0,
    epochs=args.epochs,
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    dab_weight=0.0,
    lr=args.lr,
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
mask_ratio = config.mask_ratio
mask_value = "auto"
include_zero_gene = config.include_zero_gene
max_seq_len = config.max_seq_len
n_bins = config.n_bins

input_style = "log1p"
output_style = "binned"
MLM = False; CLS = True; ADV = False; CCE = False
MVC = config.MVC
ECS = config.ecs_thres > 0
DAB = False; INPUT_BATCH_LABELS = False
input_emb_style = "continuous"
cell_emb_style = "cls"
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight
num_batch_types = None
explicit_zero_prob = MLM and include_zero_gene
do_sample_in_train = False and explicit_zero_prob
per_seq_batch_sample = False

lr = config.lr
batch_size = config.batch_size
epochs = config.epochs
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"
embsize = config.layer_size
d_hid   = config.layer_size
nlayers = config.nlayers
nhead   = config.nhead
dropout = config.dropout
log_interval = 100
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
    save_dir = Path(f"{os.getcwd()}/save/dev_{model_name}_glen{gene_panel_length}_unfix{config.unfix_last_layer}_lr{config.lr}_npos{config.n_pos}_nneg{config.n_neg}_t{config.temp}_maxl{max_seq_len}-{time.strftime('%b%d-%H-%M')}/")
else:
    save_dir = Path(f"{os.getcwd()}/save/dev_{model_name}_glen{gene_panel_length}_unfix{config.unfix_last_layer}_lr{config.lr}_npos{config.n_pos}_nneg{config.n_neg}_t{config.temp}_maxl{max_seq_len}-SuContinue/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

data_is_raw = False
model_config_file = Path(scgpt_model_path) / "args.json"
vocab_file        = Path(scgpt_model_path) / "vocab.json"

vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

train_adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in train_adata.var["gene_name"]]
gene_ids_in_vocab = np.array(train_adata.var["id_in_vocab"])
logger.info(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}.")
train_adata = train_adata[:, train_adata.var["id_in_vocab"] >= 0]

val_adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in val_adata.var["gene_name"]]
val_adata = val_adata[:, val_adata.var["id_in_vocab"] >= 0]

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
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
else:
    model_file = Path(config.load_model) / "best_model.pt"
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
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
        "gene_ids": pos_tok["genes"], "values": masked_pos,
        "target_values": pos_tok["values"], "cell_ids": list(adata_pos.obs_names),
    }
    neg_data_pt = {
        "gene_ids": neg_tok["genes"], "values": masked_neg,
        "target_values": neg_tok["values"], "cell_ids": list(adata_neg.obs_names),
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
            pos_indices = [i for i, cid in enumerate(self.positive_data["cell_ids"]) if cid != anchor_id]
            pos_samples = random.sample(pos_indices, min(self.n_pos, len(pos_indices)))
            sample.extend([{"gene_ids": self.positive_data["gene_ids"][i],
                             "values":  self.positive_data["values"][i]} for i in pos_samples])
            neg_indices = list(range(len(self.negative_data["cell_ids"])))
            neg_samples = random.sample(neg_indices, min(self.n_neg, len(neg_indices)))
            sample.extend([{"gene_ids": self.negative_data["gene_ids"][i],
                             "values":  self.negative_data["values"][i]} for i in neg_samples])
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


def train(mdl, params, epc, train_data_loader, opt, dev, dt, loss_fn):
    mdl.train()
    total_loss = 0.
    start_time = time.time()
    for batch, data in enumerate(train_data_loader):
        metrics_to_log = {}
        opt.zero_grad()
        embeddings = torch.zeros((params.batch_size, 1 + params.n_pos + params.n_neg, params.out_dim))
        for i in range(params.batch_size):
            gene_ids_ = data['gene_ids'][i].to(dev)
            values_   = data['values'][i].to(dev)
            src_key_padding_mask_ = gene_ids_.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = mdl(
                    gene_ids_, values_,
                    src_key_padding_mask=src_key_padding_mask_,
                    batch_labels=None,
                    CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS,
                    do_sample=do_sample_in_train,
                )
            embeddings[i] = output_dict["cls_output"]
        loss, a_pos_sim, a_neg_sim = loss_fn(embeddings, params.temp, params.n_pos, params.n_neg)
        loss.backward()
        opt.step()
        metrics_to_log = {
            "train/supconloss": loss.item(),
            "train/anchor_pos_sim": a_pos_sim.item(),
            "train/anchor_neg_sim": a_neg_sim.item(),
        }
        wandb.log(metrics_to_log)
        total_loss += loss.item()
        if params.verbose:
            ms_per_batch = (time.time() - start_time) * 1000
            print(f'| epoch {epc:3d} | {batch:5d}/{len(train_data_loader):5d} batches | '
                  f'lr {params.lr:02.4f} | ms/batch {ms_per_batch:6.4f} | loss {total_loss:5.2f}')
            start_time = time.time()
    return total_loss / (batch + 1)


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


def load_checkpoint(mdl, optim, filename=None):
    if filename is None:
        filename = save_dir / 'latest_ck.pt'
    ckpt = torch.load(filename, map_location='cpu' if not torch.cuda.is_available() else None)
    mdl.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    start_e = ckpt['epoch'] + 1
    best_loss = ckpt.get('best_loss', float('inf'))
    if 'best_mdl_state_dict' in ckpt:
        best_model = copy.deepcopy(mdl)
        best_model.load_state_dict(ckpt['best_mdl_state_dict'])
    else:
        best_model = None
    best_model_epoch = ckpt.get('best_model_epoch', None)
    return mdl, optim, start_e, best_loss, best_model, best_model_epoch


def evaluate(cfg, mdl, params, epc, val_data, val_data_loader, train_data, dev, dt,
             loss_fn, actual_val_data, out_prefix, ret_loss):
    mdl.eval()
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

        for pref_, obsm_key in zip(["projector", "scgpt"], ["X_scGPT_prj", "X_scGPT_trained"]):
            class_centers          = get_cluster_center(train_data_, obsm_key, False)
            class_centers_norm_val = get_cluster_center(actual_val_data_, obsm_key, True)

            train_embeddings_np      = train_data_.obsm[obsm_key]
            val_embeddings_np        = val_data_.obsm[obsm_key]
            actual_val_embeddings_np = actual_val_data_.obsm[obsm_key]

            train_labels_np      = train_data_.obs["reactivity"].values
            val_labels_np        = val_data_.obs["reactivity"].values
            actual_val_labels_np = actual_val_data_.obs["reactivity"].values

            if pref_ == "projector":
                pro_class_centers_distance_val = np.linalg.norm(class_centers_norm_val['0'] - class_centers_norm_val['1'])
                pro_ch_index = calinski_harabasz_score(val_embeddings_np, val_labels_np)
            else:
                pre_class_centers_distance_val = np.linalg.norm(class_centers_norm_val['0'] - class_centers_norm_val['1'])
                pre_ch_index = calinski_harabasz_score(val_embeddings_np, val_labels_np)

            for n in range(1, 11):
                predictions.append(knn_classifier(n, train_embeddings_np, train_labels_np, val_embeddings_np))
            predictions.append(nearest_center(val_embeddings_np, class_centers))
            predictions.append(cosine_similarity_classifier(train_embeddings_np, val_embeddings_np, train_labels_np, unique_labels))

            pro_Youdensj_threshold, pro_Youdensj_predictions, pro_jenks_threshold, pro_jenks_predictions, pro_distance_prediction_score = \
                distance_based_prediction(val_embeddings_np, class_centers, val_labels_np, actual_val_embeddings_np, actual_val_labels_np)

            metrics_to_log.update({f"{out_prefix}/{pref_}_Youdensj_threshold": pro_Youdensj_threshold})
            metrics_to_log.update({f"{out_prefix}/{pref_}_jenks_threshold": pro_jenks_threshold})

            predictions.append((pro_Youdensj_predictions, pro_distance_prediction_score))
            predictions.append((pro_jenks_predictions, pro_distance_prediction_score))
            predictions.append((pro_jenks_predictions, pro_distance_prediction_score))

            function_names += (
                [pref_ + '_knn_' + str(n) for n in range(1, 11)] +
                [pref_ + "_nearest_center", pref_ + "_cosine_similarity",
                 pref_ + "_distance_Youdensj", pref_ + "_distance_jenks", pref_ + "_svm"]
            )

        assert len(predictions) == len(function_names)

        output_file = save_dir / f"{out_prefix}_prediction_scores_e{epc}.csv"
        results_list = output_metrics(output_file, predictions, function_names, val_labels_np)

        def _r(idx): return results_list[idx]
        pro_knn_7_f1 = _r(6)[2]; pro_knn_7_auc = _r(6)[3]; pro_knn_7_ari = _r(6)[6]
        pro_nc_f1  = _r(10)[2]; pro_nc_auc  = _r(10)[3]; pro_nc_ari  = _r(10)[6]

        metrics_to_log.update({
            "epoch": epc,
            f"{out_prefix}/pro_knn_7_f1":  pro_knn_7_f1,
            f"{out_prefix}/pro_knn_7_auc": pro_knn_7_auc,
            f"{out_prefix}/pro_knn_7_ari": pro_knn_7_ari,
            f"{out_prefix}/pro_nc_f1":     pro_nc_f1,
            f"{out_prefix}/pro_nc_auc":    pro_nc_auc,
            f"{out_prefix}/pro_nc_ari":    pro_nc_ari,
            f"{out_prefix}/pro_ch_index":  pro_ch_index,
            f"{out_prefix}/projector_center_distance": pro_class_centers_distance_val,
            f"{out_prefix}/scgpt_center_distance":     pre_class_centers_distance_val,
        })

        pro_output_figure = save_dir / f"{out_prefix}_proj_embeddings_reactivity_umap_e{epc}.png"
        with plt.rc_context({"figure.figsize": (10, 2), "figure.dpi": (100)}):
            sc.pl.umap(val_data_, color=["reactivity", 'pro_knn_7' if 'pro_knn_7' in val_data_.obs else "reactivity"],
                       show=False)
            plt.savefig(pro_output_figure, dpi=100)

        if out_prefix == 'testing':
            metrics_to_log[f"{out_prefix}/scGPT_prj_cell_umap"] = wandb.Image(
                str(pro_output_figure),
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
                                  batch_labels=None, CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS,
                                  do_sample=do_sample_in_train)
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

        wandb.log(metrics_to_log)

    if ret_loss:
        return averaged_loss, pro_class_centers_distance_val, pre_class_centers_distance_val
    else:
        return pro_class_centers_distance_val, pre_class_centers_distance_val


def main(model):
    torch.backends.cudnn.benchmark = True
    lr, epochs = args.lr, args.epochs
    print('==> device:', device, '| dtype:', dtype, '\n==> args:', args)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8)
    criterion = SupConHardLoss
    best_loss = float('inf')
    best_model = None
    define_wandb_metrcis()

    val_pos_data_pt, val_neg_data_pt = prepare_data(val_pos_tokenized, val_neg_tokenized, val_adata_pos, val_adata_neg)
    val_dataset = CustomDataset(val_pos_data_pt, val_neg_data_pt, args.n_pos, args.n_neg)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, drop_last=True)

    start_epoch = 1
    checkpoint_file = save_dir / 'latest_ck.pt'
    if os.path.exists(checkpoint_file):
        print("Resuming from checkpoint...")
        model, optimizer, start_epoch, best_loss, best_model, best_model_epoch = \
            load_checkpoint(model, optimizer, checkpoint_file)
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        pos_data_pt, neg_data_pt = prepare_data(train_pos_tokenized, train_neg_tokenized, train_adata_pos, train_adata_neg)
        dataset      = CustomDataset(pos_data_pt, neg_data_pt, args.n_pos, args.n_neg)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=True)

        t_epoch = time.time()
        train_loss = train(model, args, epoch, train_loader, optimizer, device, dtype, criterion)
        t_train = time.time() - t_epoch

        val_loss, val_pro_dist, val_pre_dist = evaluate(
            config, model, args, epoch, val_adata, val_loader, train_adata,
            device, dtype, criterion, val_adata, "validation", True
        )
        elapsed = time.time() - t_epoch

        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed/60:.2f}m | train: {t_train/60:.2f}m | "
            f"valid loss {val_loss:.4f} | pro_dist {val_pro_dist:.2f}"
        )
        logger.info("-" * 89)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best from epoch : {epoch:3d}; loss: {best_loss:6.4f}")

        ckpt_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'best_loss':  best_loss,
            'best_mdl_state_dict': best_model.state_dict(),
            'best_model_epoch':    best_model_epoch,
        }
        save_checkpoint(ckpt_state)

        if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
            save_checkpoint(ckpt_state, filename=save_dir / f'model_e{epoch}.pt')
            evaluate(config, model, args, epoch, test_adata, val_loader, train_adata,
                     device, dtype, criterion, val_adata, "testing", False)

    evaluate(config, best_model, args, best_model_epoch, test_adata, val_loader, train_adata,
             device, dtype, criterion, val_adata, "testing", False)
    logger.info(f"Best epoch: {best_model_epoch}")

    run.finish()
    wandb.finish()
    gc.collect()


if __name__ == '__main__':
    main(model)
