# scTRP — Single-cell T/B-cell Reactivity Prediction

`scTRP` is a Python package for predicting T cell reactivity from single-cell multimodal data (scRNA-seq + scTCR-seq sequences). It offers two tiers:

- **Full pipeline** — supervised contrastive learning on top of [scGPT](https://github.com/bowang-lab/scGPT) gene-expression embeddings and [ESM2](https://github.com/facebookresearch/esm) TCR-sequence embeddings (GPU required).
- **Light alternative** — an XGBoost classifier trained directly on a small gene panel; no GPU, scGPT, or ESM2 needed.

---

## Table of Contents

1. [Installation](#installation)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
   - [Mode A — SimCLR + SupCon-Hard (ESM2 only)](#mode-a--simclr--supcon-hard-esm2-only)
   - [Mode B — SimCLR + SupCon-Hard (ESM2 + extra features)](#mode-b--simclr--supcon-hard-esm2--extra-features)
   - [Mode C — SupCon-Hard only (no TCR sequence branch)](#mode-c--supcon-hard-only-no-tcr-sequence-branch)
   - [Mode D — XGBoost light classifier (no GPU required)](#mode-d--xgboost-light-classifier-no-gpu-required)
5. [Inference (full pipeline)](#inference-full-pipeline)
6. [Package API](#package-api)
7. [Repository Layout](#repository-layout)

---

## Installation

### From source (recommended)

```bash
git clone https://github.com/LiSu935/scTRP.git
cd scTRP
pip install -e .
```

### From PyPI (once published)

```bash
pip install scTRP
```

### Light XGBoost alternative (no GPU)

```bash
pip install -e ".[light]"
# installs: xgboost, scanpy, joblib, matplotlib
```

### Full training pipeline (GPU required)

Training requires additional heavy dependencies (wandb, webdataset, ESM2, scGPT, flash-attn):

```bash
pip install -e ".[train]"
```

> **scGPT** must be installed separately from source:
> ```bash
> git clone https://github.com/bowang-lab/scGPT
> pip install -e /path/to/scGPT
> ```
> The scripts auto-detect scGPT at a few known cluster paths; set the path manually in
> `scTRP/inference/functions.py` if needed.

---

## Environment Setup

Tested with Python 3.10 + CUDA 11.7:

```bash
conda create -n sctrp python=3.10
conda activate sctrp

# PyTorch (adjust cuda version to match your driver)
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 \
    -c pytorch -c nvidia

# Bioinformatics stack
conda install -c conda-forge matplotlib pandas scikit-learn scipy
pip install tqdm PyYAML webdataset fair-esm anndata muon scanpy scirpy
conda install -c huggingface transformers==4.32.1
pip install peft wandb "flash-attn<1.0.5" pot

# scGPT (from source)
git clone https://github.com/bowang-lab/scGPT /path/to/scGPT
pip install -e /path/to/scGPT

# scTRP
pip install -e /path/to/scTRP
```

---

## Data Preparation

### Step 1 — Build AnnData from 10x output

```bash
python data_prep/load_TCR_RNA_prep.py \
    --input  /path/to/10x_dir \
    --output /path/to/output.h5ad
```

Output `.h5ad` contains:
- `adata.X` — CPM-normalized, log1p-transformed expression (2000 HVGs)
- `adata.obs['reactivity']` — binary label (`'0'` / `'1'`)
- `adata.obs['cdr3_aa']` — paired TRA+TRB CDR3 sequence (separated by `;`)

### Step 2 — Encode CDR3 sequences with ESM2

Encodes CDR3 sequences offline and saves `.npz` files inside a webdataset `.tar` archive.
Each sample gets an `esm2_emb` key (shape `[1, 1280]`).

```bash
python classifier/data_prep/encode_seq_with_pretrained_ESM2.py \
    --input_h5ad  /path/to/data.h5ad \
    --output_tar  /path/to/encoded.tar
```

### Step 2b — Encode with extra features (optional)

If you have per-cell extra features (e.g. HMM similarity scores), use the extra-feat variant.
Features are z-score normalized using **train-set statistics**; missing values are filled with 0
before statistics are computed.

```bash
python classifier/data_prep/encode_seq_with_pretrained_ESM2_extraFeat.py \
    --input_h5ad       /path/to/data.h5ad \
    --output_tar       /path/to/encoded_extrafeat.tar \
    --extra_feat_keys  hmm_score_tra hmm_score_trb \
    --train_h5ad       /path/to/train.h5ad
```

---

## Training

### Mode A — SimCLR + SupCon-Hard (ESM2 only)

Joint training of a scGPT GEX encoder (`LayerNormNet`) and an ESM2 TCR encoder (`MoBYMLP`)
with both SimCLR cross-modal alignment and Supervised Contrastive-Hard loss.

```bash
python scripts/train_simclr.py \
    --train_data_path        /path/to/train.h5ad \
    --val_data_path          /path/to/val.h5ad \
    --test_data_path         /path/to/test.h5ad \
    --train_data_path_simclr /path/to/train_esm2encoded.tar \
    --val_data_path_simclr   /path/to/val_esm2encoded.tar \
    --model_name             MyExperiment \
    --lr 1e-4  --lr_seq 1e-4 \
    --epochs 100  --batch_size 32 \
    --n_pos 9  --n_neg 30  --temp 0.1 \
    --simclr_weight 1.0 \
    --out_dim 128  --hidden_dim 256 \
    --unfix_last_layer 1
```

### Mode B — SimCLR + SupCon-Hard (ESM2 + extra features)

Same as Mode A but appends N extra features to the ESM2 embedding before the `MoBYMLP`
projector. Set `--extra_feat_dim` to the number of extra features.

```bash
python scripts/train_simclr_extrafeat.py \
    --train_data_path        /path/to/train.h5ad \
    --val_data_path          /path/to/val.h5ad \
    --test_data_path         /path/to/test.h5ad \
    --train_data_path_simclr /path/to/train_extrafeat.tar \
    --val_data_path_simclr   /path/to/val_extrafeat.tar \
    --extra_feat_dim 2 \
    --model_name             MyExperiment_extraFeat \
    --lr 1e-4  --lr_seq 1e-4 \
    --epochs 100  --batch_size 32 \
    --n_pos 9  --n_neg 30  --temp 0.1 \
    --simclr_weight 1.0 \
    --out_dim 128  --hidden_dim 256 \
    --unfix_last_layer 1
```

> Setting `--extra_feat_dim 0` is identical to Mode A.

### Mode C — SupCon-Hard only (no TCR sequence branch)

Trains only the scGPT GEX encoder (`LayerNormNet`) with Supervised Contrastive-Hard loss.
No webdataset `.tar` files are needed.

```bash
python scripts/train_supcon.py \
    --train_data_path /path/to/train.h5ad \
    --val_data_path   /path/to/val.h5ad \
    --test_data_path  /path/to/test.h5ad \
    --model_name      MyExperiment_SupCon \
    --lr 1e-4 \
    --epochs 100  --batch_size 32 \
    --n_pos 9  --n_neg 30  --temp 0.1 \
    --out_dim 128  --hidden_dim 256 \
    --unfix_last_layer 1
```

### Mode D — XGBoost light classifier (no GPU required)

Trains an XGBoost binary classifier directly on a small curated gene panel extracted from
`.h5ad` files. No ESM2 encoding, no TCR sequences, no GPU needed. Outputs per-cell predicted
labels and reactive scores to CSV files alongside the saved model.

```bash
python scripts/train_xgboost.py \
    --train_data_path /path/to/train.h5ad \
    --val_data_path   /path/to/val.h5ad \
    --test_data_path  /path/to/test.h5ad \
    --output_dir      /path/to/output/
```

To use a custom gene list instead of the built-in 30-gene panel:

```bash
python scripts/train_xgboost.py \
    --train_data_path /path/to/train.h5ad \
    --val_data_path   /path/to/val.h5ad \
    --test_data_path  /path/to/test.h5ad \
    --output_dir      /path/to/output/ \
    --gene_list       /path/to/genes.csv
```

The CSV should have gene names in the first column (no header required, or any header).

**Outputs written to `--output_dir`:**

| File | Description |
|---|---|
| `xgboost_reactivity.json` | XGBoost model (native format) |
| `xgboost_reactivity.pkl` | XGBoost model (joblib format) |
| `gene_importance.csv` | Feature importance for each gene |
| `feature_importance.png` | Bar chart of top-30 gene importances |
| `metrics_test.csv` | Test-set metrics (AUROC, F1, MCC, etc.) |
| `predictions_train.csv` | Per-cell predicted label + reactive score (train) |
| `predictions_val.csv` | Per-cell predicted label + reactive score (val) |
| `predictions_test.csv` | Per-cell predicted label + reactive score (test) |

**XGBoost hyperparameters (all have defaults matching the validated notebook):**

| Parameter | Description | Default |
|---|---|---|
| `--n_estimators` | Max number of trees | `500` |
| `--learning_rate` | Boosting learning rate | `0.05` |
| `--max_depth` | Max tree depth | `4` |
| `--min_child_weight` | Min samples per leaf | `5` |
| `--subsample` | Row subsampling per tree | `0.8` |
| `--colsample_bytree` | Feature subsampling per tree | `0.8` |
| `--gamma` | Min loss reduction to split | `1.0` |
| `--reg_alpha` | L1 regularisation | `0.1` |
| `--reg_lambda` | L2 regularisation | `1.0` |
| `--early_stopping_rounds` | Stop if val AUC stagnates | `30` |

---

### Key Tunable Parameters (full pipeline)

| Parameter | Script | Description | Default |
|---|---|---|---|
| `--lr` | all | Learning rate for scGPT encoder | `1e-4` |
| `--lr_seq` | A / B | Learning rate for MoBYMLP (TCR branch) | `1e-4` |
| `--epochs` | all | Number of training epochs | `100` |
| `--batch_size` | all | Batch size | `32` |
| `--n_pos` | all | Positives per anchor in SupCon-Hard | `9` |
| `--n_neg` | all | Negatives per anchor in SupCon-Hard | `30` |
| `--temp` | all | Temperature for SupCon-Hard loss | `0.1` |
| `--temperature` | A / B | Temperature for SimCLR info-NCE | `0.05` |
| `--simclr_weight` | A / B | Weight of SimCLR loss relative to SupCon loss | `1.0` |
| `--out_dim` | all | Output embedding dimension | `128` |
| `--hidden_dim` | all | Hidden dim of LayerNormNet | `256` |
| `--unfix_last_layer` | all | Number of last scGPT layers to unfreeze | `0` |
| `--extra_feat_dim` | B | Number of extra features appended to ESM2 | `0` |
| `--warmup` | all | LR warmup steps | `50` |
| `--save_eval_interval` | all | Epochs between checkpoint saves | `10` |
| `--load_local_pretrain` | all | Path to local scGPT pretrained checkpoint dir | `None` |

---

## Inference (full pipeline)

Runs trained checkpoints on a test cohort. Supports both in-distribution evaluation
(AUROC / accuracy metrics computed against ground-truth labels) and new-data mode
(OT-based reactivity scoring for unlabeled cohorts).

```bash
python scripts/infer.py \
    --train_data_path             /path/to/train.h5ad \
    --test_data_path              /path/to/test.h5ad \
    --load_local_pretrain_pathList /path/to/model1.pt,/path/to/model2.pt \
    --batch_size 32
```

For new (unlabeled) cohort scoring, add `--test_new_data`:

```bash
python scripts/infer.py \
    --train_data_path             /path/to/train.h5ad \
    --test_data_path              /path/to/new_cohort.h5ad \
    --load_local_pretrain_pathList /path/to/model.pt \
    --test_new_data
```

### Inference Parameters

| Parameter | Description | Default |
|---|---|---|
| `--load_local_pretrain_pathList` | Comma-separated list of `.pt` checkpoint paths | required |
| `--train_data_path` | Training `.h5ad` (builds reference embeddings) | required |
| `--test_data_path` | Test `.h5ad` | required |
| `--batch_size` | Batch size for embedding extraction | `32` |
| `--max_seq_len` | Max gene sequence length for scGPT tokenizer | `1200` |
| `--test_new_data` | Switch to OT-based new-cohort mode | `False` |

---

## Package API

The `scTRP` package exposes the core building blocks for programmatic use:

```python
from scTRP.models.layers import LayerNormNet, MoBYMLP
from scTRP.losses import SupConHardLoss
from scTRP.training.simclr import SimCLR, SimCLRWithExtraFeat, build_seq_input
from scTRP.inference.functions import (
    get_project_emb, get_cluster_center,
    knn_classifier, OT_based_prediction, output_metrics,
)
```

### `LayerNormNet(hidden_dim, out_dim, drop_out=0.1)`

Three-layer MLP with LayerNorm. Acts as the scGPT CLS-token decoder / projector head.
Input dim is fixed at 512 (scGPT CLS token size).

### `MoBYMLP(in_dim=1280, inner_dim=2048, out_dim=128)`

Two-layer projector MLP (BatchNorm + ReLU). Used as the ESM2 TCR-sequence projector.
When `extra_feat_dim > 0`, set `in_dim = 1280 + extra_feat_dim`.

### `SupConHardLoss(model_emb, temp, n_pos, n_neg)`

Supervised Contrastive-Hard loss with explicit anchor/positive/negative sampling.
Returns `(loss, pos_sim_mean, neg_sim_mean)`.

### `build_seq_input(batch, extra_feat_dim, device) -> Tensor`

Concatenates the ESM2 CLS token (1280-dim) with optional extra features from a batch dict.
Returns shape `(bsz, 1280 + extra_feat_dim)`. No-op when `extra_feat_dim=0`.

### `OT_based_prediction(train_data, test_data, ...)`

Optimal-Transport-based reactivity scoring for new cohorts without ground-truth labels.

---

## Repository Layout

```
scTRP/                         # pip-installable package
├── __init__.py
├── models/
│   └── layers.py              # LayerNormNet, MoBYMLP
├── losses.py                  # SupConHardLoss
├── training/
│   └── simclr.py              # SimCLR, SimCLRWithExtraFeat, build_seq_input
└── inference/
    └── functions.py           # KNN, OT, AUROC helpers

scripts/                       # entry-point scripts (run directly with python)
├── train_simclr.py            # Mode A: SimCLR + SupCon (ESM2 only)
├── train_simclr_extrafeat.py  # Mode B: SimCLR + SupCon + extra features
├── train_supcon.py            # Mode C: SupCon-Hard only
├── train_xgboost.py           # Mode D: XGBoost light classifier (no GPU)
└── infer.py                   # Inference / evaluation (full pipeline)

classifier/                    # original scripts (reference implementations)
├── data_prep/                 # ESM2 encoding, HMM feature prep
├── case_study/                # ICB cohort evaluation (Yost, Caushi)
├── model_explainability/      # Attention weights, GRN inference
├── utils/                     # Shared utilities
└── archived/                  # Superseded scripts (kept for reference)

data_prep/                     # Raw 10x → AnnData conversion
pyproject.toml
requirements.txt
README.md
```
