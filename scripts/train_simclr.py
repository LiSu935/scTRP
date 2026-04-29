"""
scripts/train_simclr.py  —  SimCLR + SupCon-Hard training (ESM2-only, no extra features).

This script is functionally equivalent to:
    python train_simclr_extrafeat.py --extra_feat_dim 0 ...

Use train_simclr_extrafeat.py if you want to append HMM scores or other
per-cell features to the ESM2 embedding.

EXAMPLE COMMAND:
    python scripts/train_simclr.py \\
        --train_data_path  /path/to/train.h5ad \\
        --val_data_path    /path/to/val.h5ad \\
        --test_data_path   /path/to/test.h5ad \\
        --train_data_path_simclr  /path/to/train_esm2encoded.tar \\
        --val_data_path_simclr    /path/to/val_esm2encoded.tar \\
        --lr 1e-4  --lr_seq 1e-4  --epochs 100  --batch_size 32 \\
        --n_pos 9  --n_neg 30  --temp 0.1  --simclr_weight 1.0 \\
        --unfix_last_layer 1  --out_dim 128  --hidden_dim 256 \\
        --model_name MyExperiment
"""

import sys

# Enforce extra_feat_dim=0 so this script behaves as the ESM2-only version.
# If the user already passed --extra_feat_dim, respect their value.
if '--extra_feat_dim' not in sys.argv:
    sys.argv += ['--extra_feat_dim', '0']

# Delegate all logic to train_simclr_extrafeat.py (same file, just with
# extra_feat_dim forced to 0, which disables the extra-feature pathway).
import importlib.util, os

_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_simclr_extrafeat.py")
spec = importlib.util.spec_from_file_location("train_simclr_extrafeat", _script)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
