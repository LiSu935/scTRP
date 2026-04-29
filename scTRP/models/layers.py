"""
Core neural network layers shared across training and inference.

LayerNormNet  — scGPT projector head (cls_decoder).
MoBYMLP       — MoBY-style MLP projector for ESM2/extra-feat embeddings.
"""

import torch
import torch.nn as nn


class LayerNormNet(nn.Module):
    """Three-layer MLP with LayerNorm, used as the scGPT cls_decoder."""

    def __init__(self, hidden_dim: int, out_dim: int, drop_out: float = 0.1):
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
        x = self.fc3(x)
        return x


class MoBYMLP(nn.Module):
    """MoBY-style MLP projector.

    When used with ESM2 embeddings: in_dim = 1280 (ESM2-650M CLS-token dim).
    When extra features are appended: in_dim = 1280 + extra_feat_dim.
    out_dim (default 128) is kept fixed so downstream losses are unaffected.
    """

    def __init__(
        self,
        in_dim: int = 256,
        inner_dim: int = 4096,
        out_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = (
            nn.Linear(in_dim if num_layers == 1 else inner_dim, out_dim)
            if num_layers >= 1
            else nn.Identity()
        )

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return x
