"""
scTRP — single-cell T-cell Reactivity Prediction.

A pip-installable toolkit for contrastive learning-based TCR reactivity
classification using scGPT (gene expression) and ESM2 (CDR3 sequence).
"""

__version__ = "0.1.0"

from scTRP.models.layers import LayerNormNet, MoBYMLP
from scTRP.losses import SupConHardLoss
from scTRP.training.simclr import (
    AverageMeter,
    log_negative_mean_logtis,
    SimCLR,
    build_seq_input,
    SimCLRWithExtraFeat,
)

__all__ = [
    "LayerNormNet",
    "MoBYMLP",
    "SupConHardLoss",
    "AverageMeter",
    "log_negative_mean_logtis",
    "SimCLR",
    "build_seq_input",
    "SimCLRWithExtraFeat",
]
