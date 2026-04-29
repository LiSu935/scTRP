"""
Loss functions for contrastive training.

SupConHardLoss — Supervised Contrastive loss with explicit anchor/pos/neg triplets.
Reference: https://github.com/tttianhao/CLEAN/blob/main/app/src/CLEAN/losses.py
"""

import torch
import torch.nn.functional as F


def SupConHardLoss(model_emb, temp: float, n_pos: int, n_neg: int):
    """Supervised Contrastive-Hard loss.

    Args:
        model_emb: Tensor of shape (bsz, 1 + n_pos + n_neg, out_dim).
                   Axis-1 layout: [anchor, n_pos positives, n_neg negatives].
        temp:      Temperature scalar.
        n_pos:     Number of positive examples per anchor.
        n_neg:     Number of negative examples per anchor.

    Returns:
        loss          — scalar loss.
        pos_sim_mean  — mean anchor-positive similarity (for logging).
        neg_sim_mean  — mean anchor-negative similarity (for logging).
    """
    features = F.normalize(model_emb, dim=-1, p=2)
    features_T = torch.transpose(features, 1, 2)
    anchor = features[:, 0]
    anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T) / temp
    anchor_dot_features = anchor_dot_features.squeeze(1)
    logits = anchor_dot_features - 1 / temp
    exp_logits = torch.exp(logits[:, 1:])
    exp_logits_sum = n_pos * torch.log(exp_logits.sum(1))
    pos_logits_sum = logits[:, 1 : n_pos + 1].sum(1)
    log_prob = (pos_logits_sum - exp_logits_sum) / n_pos
    loss = -log_prob.mean()
    return loss, (pos_logits_sum / n_pos).mean(), (logits[:, n_pos + 1 :].sum(1) / n_neg).mean()
