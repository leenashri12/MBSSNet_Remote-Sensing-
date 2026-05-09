"""
VSS Block — Visual State Space Block
======================================
Layer Norm → branch split:
  - branch a: Linear → DW-Conv → SiLU → SS2D → LayerNorm
  - branch b: Linear → SiLU (gate)
  merge: a * b → Linear → residual
"""

import torch
import torch.nn as nn
from .ss2d import SS2D


class VSSBlock(nn.Module):
    """
    Single VSS block as shown in the paper's architecture diagram.

    Parameters
    ----------
    hidden_dim : channel dimension
    d_state    : SSM state size (default 16)
    d_conv     : DW-conv kernel in SS2D (default 3)
    expand     : expansion ratio in SS2D (default 2)
    dropout    : dropout rate
    """

    def __init__(self, hidden_dim, d_state=16, d_conv=3, expand=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ss2d = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, x):
        """
        x : (B, C, H, W)  — channel-first (standard conv format)
        returns : (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x

        # channel-last for LayerNorm + SS2D
        x = x.permute(0, 2, 3, 1).contiguous()   # (B, H, W, C)
        x = self.norm(x)
        x = self.ss2d(x)                          # (B, H, W, C)

        # back to channel-first + residual
        x = x.permute(0, 3, 1, 2).contiguous()    # (B, C, H, W)
        return residual + x
