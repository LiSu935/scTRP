"""
SimCLR training utilities.

AverageMeter        — running-average tracker.
log_negative_mean_logtis — helper for logging off-diagonal logit means.
save_config_file    — saves args to a YAML in the checkpoint folder.
SimCLR              — cross-modal info-NCE training/validation loop.
build_seq_input     — concatenate ESM2 embedding with optional extra features.
SimCLRWithExtraFeat — SimCLR subclass whose validation() passes extra_feats.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import yaml


# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #

class AverageMeter:
    """Computes and stores the running average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_negative_mean_logtis(logits, mode: str, minibatch_size: int) -> float:
    N = minibatch_size
    if mode == "struct_struct":
        return logits[0:N, 1:N].mean().item()
    if mode == "struct_seq":
        return logits[0:N, N : 2 * N - 1].mean().item()
    if mode == "seq_struct":
        return logits[N : 2 * N, 1:N].mean().item()
    if mode == "seq_seq":
        return logits[N : 2 * N, N : 2 * N - 1].mean().item()


def save_config_file(model_checkpoints_folder: str, args) -> None:
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    with open(os.path.join(model_checkpoints_folder, "config.yml"), "w") as f:
        yaml.dump(args, f, default_flow_style=False)


# ------------------------------------------------------------------ #
# Extra-feature helper
# ------------------------------------------------------------------ #

def build_seq_input(batch, extra_feat_dim: int, device) -> torch.Tensor:
    """Concatenate ESM2 CLS-token embedding with optional z-scored extra features.

    Args:
        batch:          List of dicts from a webdataset .npz.  Each dict must
                        contain 'esm2_emb' (shape (1, 1280)) and, when
                        extra_feat_dim > 0, 'extra_feats' (shape (1, N)).
        extra_feat_dim: Number of extra features to append (0 = disabled).
        device:         Target torch device.

    Returns:
        Tensor of shape (bsz, 1280 + extra_feat_dim) on `device`.
    """
    esm2 = np.concatenate([d["esm2_emb"] for d in batch], axis=0)  # (bsz, 1280)

    if extra_feat_dim > 0:
        extra_list = []
        for d in batch:
            ef = d["extra_feats"]
            ef = ef.reshape(1, -1) if ef.ndim == 1 else ef
            extra_list.append(ef)
        extra = np.concatenate(extra_list, axis=0)  # (bsz, N)
        if extra.shape[1] != extra_feat_dim:
            raise ValueError(
                f"--extra_feat_dim={extra_feat_dim} but .npz 'extra_feats' has "
                f"{extra.shape[1]} column(s). Re-run encode_seq_with_pretrained_ESM2_extraFeat.py."
            )
        combined = np.concatenate([esm2, extra], axis=-1)
    else:
        combined = esm2

    return torch.from_numpy(combined).float().to(device)


# ------------------------------------------------------------------ #
# SimCLR
# ------------------------------------------------------------------ #

class SimCLR(nn.Module):
    """Cross-modal SimCLR training/validation loop (scGPT ↔ ESM2 MoBYMLP).

    Keyword arguments mirror the original classifier script.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = kwargs["args"]
        self.config = kwargs["config"]
        self.device = kwargs["device"]
        self.model_gex = kwargs["model_gex"].to(self.device)
        self.model_seq = kwargs["model_seq"].to(self.device)
        self.optimizer_seq = kwargs["optimizer_seq"]
        self.optimizer_gex = kwargs["optimizer_gex"]
        self.vocab = kwargs["vocab"]
        self.pad_token = kwargs["pad_token"]
        self.pad_value = kwargs["pad_value"]
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.batch_seq_mixlen: list = []
        self.batch_seq_samelen: list = []
        self.images_mixlen: list = []
        self.images_samelen: list = []

    def info_nce_loss(self, features_gex, features_seq):
        labels = torch.cat(
            [torch.arange(self.args.batch_size) for _ in range(self.args.n_views)], dim=0
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.device)
        features_gex = F.normalize(features_gex, dim=1)
        features_seq = F.normalize(features_seq, dim=1)
        features = torch.cat([features_gex, features_seq], dim=0)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.args.temperature
        return logits, labels

    def reset_current_samelen_batch(self):
        self.batch_seq_samelen = []
        self.images_samelen = []

    def check_samelen_batch(self, batch_size: int) -> bool:
        return len(self.batch_seq_samelen) == batch_size

    def reset_current_mixlen_batch(self):
        self.batch_seq_mixlen = []
        self.images_mixlen = []

    def check_mixlen_batch(self, batch_size: int) -> bool:
        return len(self.batch_seq_mixlen) == batch_size

    @torch.no_grad()
    def validation(self, val_loader):
        loss_val_sum = AverageMeter()
        l_prob = AverageMeter()
        negsim_gex_seq = AverageMeter()
        negsim_seq_seq = AverageMeter()
        negsim_gex_gex = AverageMeter()
        self.model_gex.eval()
        self.model_seq.eval()

        for batch_data in iter(val_loader):
            batch = batch_data[0]
            bsz = len(batch)
            gex_input = [(d["genes_data"], d["expressions_data"]) for d in batch]
            batch_esm2_emb = np.concatenate([d["esm2_emb"] for d in batch], axis=0)

            batch_genes_data, batch_expressions_data = zip(*gex_input)
            max_len = max(len(g) for g in batch_genes_data)
            bg_tensor = torch.full((bsz, max_len), self.vocab[self.pad_token], dtype=torch.int64)
            be_tensor = torch.full((bsz, max_len), self.pad_value, dtype=torch.float32)
            for i, (gd, ed) in enumerate(zip(batch_genes_data, batch_expressions_data)):
                bg_tensor[i, : len(gd)] = gd
                be_tensor[i, : len(ed)] = ed

            batch_esm2_emb_tensor = torch.from_numpy(batch_esm2_emb).float().to(self.device)
            bg_tensor = bg_tensor.to(self.device)
            be_tensor = be_tensor.to(self.device)

            with autocast(enabled=self.config.amp):
                src_key_padding_mask = bg_tensor.eq(self.vocab[self.pad_token])
                features_gex = self.model_gex(
                    bg_tensor, be_tensor, src_key_padding_mask,
                    batch_labels=None, CLS=True, CCE=False,
                    MVC=self.config.MVC, ECS=self.config.ecs_thres > 0,
                    do_sample=False,
                )["cls_output"]
                features_seq = self.model_seq(batch_esm2_emb_tensor)
                logits, labels = self.info_nce_loss(features_gex, features_seq)
                loss_val_sum.update(self.criterion(logits, labels), bsz)
                l_prob.update(logits[:, 0].mean().item(), bsz)
                negsim_gex_gex.update(log_negative_mean_logtis(logits, "struct_struct", bsz), bsz)
                negsim_gex_seq.update(log_negative_mean_logtis(logits, "struct_seq", bsz), bsz)
                negsim_seq_seq.update(log_negative_mean_logtis(logits, "seq_seq", bsz), bsz)

        return (
            float(loss_val_sum.avg),
            float(l_prob.avg),
            float(negsim_gex_gex.avg),
            float(negsim_gex_seq.avg),
            float(negsim_seq_seq.avg),
        )

    def train(self, train_loader, val_loader):
        n_iter = 0
        start = time.time()
        losses = AverageMeter()
        simclr_losses = AverageMeter()
        bsz = self.args.batch_size
        self.reset_current_mixlen_batch()
        self.reset_current_samelen_batch()
        current_length = 0
        current_ratio = 0

        for batch_data in iter(train_loader):
            self.model_gex.train()
            self.model_seq.train()
            batch = batch_data[0]
            batch_esm2_emb = np.concatenate([d["esm2_emb"] for d in batch], axis=0)
            seqlen = len(str(batch[0]["seq"]))
            if len(self.batch_seq_samelen) == 0:
                current_length = seqlen

            if abs(seqlen - current_length) <= self.args.length_range:
                self.images_samelen.append((batch[0]["genes_data"], batch[0]["expressions_data"]))
                self.batch_seq_samelen.append((batch[0]["index"], str(batch[0]["seq"])))

            if self.check_samelen_batch(bsz):
                gex_input = self.images_samelen
                current_ratio = 0
            else:
                continue

            batch_esm2_emb_tensor = torch.from_numpy(batch_esm2_emb).float().to(self.device)
            batch_genes_data, batch_expressions_data = zip(*gex_input)
            max_len = max(len(g) for g in batch_genes_data)
            bg_tensor = torch.full((bsz, max_len), self.vocab[self.pad_token], dtype=torch.int64)
            be_tensor = torch.full((bsz, max_len), self.pad_value, dtype=torch.float32)
            for i, (gd, ed) in enumerate(zip(batch_genes_data, batch_expressions_data)):
                bg_tensor[i, : len(gd)] = gd
                be_tensor[i, : len(ed)] = ed
            bg_tensor = bg_tensor.to(self.device)
            be_tensor = be_tensor.to(self.device)

            with autocast(enabled=self.config.amp):
                src_key_padding_mask = bg_tensor.eq(self.vocab[self.pad_token])
                features_gex = self.model_gex(
                    bg_tensor, be_tensor, src_key_padding_mask,
                    batch_labels=None, CLS=True, CCE=False,
                    MVC=self.config.MVC, ECS=self.config.ecs_thres > 0,
                    do_sample=False,
                )["cls_output"]
                features_seq = self.model_seq(batch_esm2_emb_tensor)
                logits, labels = self.info_nce_loss(features_gex, features_seq)
                simclr_loss = self.criterion(logits, labels)
                loss = simclr_loss

            losses.update(loss.item(), bsz)
            simclr_losses.update(simclr_loss.item(), bsz)
            l_prob = logits[:, 0].mean()
            negsim_gex_gex = log_negative_mean_logtis(logits, "struct_struct", bsz)
            negsim_gex_seq = log_negative_mean_logtis(logits, "struct_seq", bsz)
            negsim_seq_seq = log_negative_mean_logtis(logits, "seq_seq", bsz)

            if self.check_samelen_batch(bsz):
                self.reset_current_samelen_batch()
            else:
                self.reset_current_mixlen_batch()

            end = time.time()
            print("one epoch cost ", end - start)


# ------------------------------------------------------------------ #
# Extra-feature subclass
# ------------------------------------------------------------------ #

class SimCLRWithExtraFeat(SimCLR):
    """Drop-in replacement for SimCLR that threads extra_feats through validation().

    Identical to SimCLR when extra_feat_dim == 0.
    """

    def __init__(self, extra_feat_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.extra_feat_dim = extra_feat_dim

    @torch.no_grad()
    def validation(self, val_loader):
        loss_val_sum = AverageMeter()
        l_prob = AverageMeter()
        negsim_gex_seq = AverageMeter()
        negsim_seq_seq = AverageMeter()
        negsim_gex_gex = AverageMeter()
        self.model_gex.eval()
        self.model_seq.eval()

        for batch_data in iter(val_loader):
            batch = batch_data[0]
            bsz = len(batch)
            gex_input = [(d["genes_data"], d["expressions_data"]) for d in batch]

            seq_input_tensor = build_seq_input(batch, self.extra_feat_dim, self.device)

            batch_genes_data, batch_expressions_data = zip(*gex_input)
            max_len = max(len(g) for g in batch_genes_data)
            bg_tensor = torch.full((bsz, max_len), self.vocab[self.pad_token], dtype=torch.int64)
            be_tensor = torch.full((bsz, max_len), self.pad_value, dtype=torch.float32)
            for i, (gd, ed) in enumerate(zip(batch_genes_data, batch_expressions_data)):
                bg_tensor[i, : len(gd)] = gd
                be_tensor[i, : len(ed)] = ed
            bg_tensor = bg_tensor.to(self.device)
            be_tensor = be_tensor.to(self.device)

            with autocast(enabled=self.config.amp):
                src_key_padding_mask = bg_tensor.eq(self.vocab[self.pad_token])
                features_gex = self.model_gex(
                    bg_tensor, be_tensor, src_key_padding_mask,
                    batch_labels=None, CLS=True, CCE=False,
                    MVC=self.config.MVC, ECS=self.config.ecs_thres > 0,
                    do_sample=False,
                )["cls_output"]
                features_seq = self.model_seq(seq_input_tensor)
                logits, labels = self.info_nce_loss(features_gex, features_seq)
                loss_val_sum.update(self.criterion(logits, labels), bsz)
                l_prob.update(logits[:, 0].mean().item(), bsz)
                negsim_gex_gex.update(log_negative_mean_logtis(logits, "struct_struct", bsz), bsz)
                negsim_gex_seq.update(log_negative_mean_logtis(logits, "struct_seq", bsz), bsz)
                negsim_seq_seq.update(log_negative_mean_logtis(logits, "seq_seq", bsz), bsz)

        return (
            float(loss_val_sum.avg),
            float(l_prob.avg),
            float(negsim_gex_gex.avg),
            float(negsim_gex_seq.avg),
            float(negsim_seq_seq.avg),
        )
