"""
CMFM — Cross-Modal Fusion Module
==================================
Fuses SAR and Optical features at each encoder scale via:
  1. Layer Norm on both modalities
  2. Channel Exchange (lightweight, parameter-free)
  3. SFFM: two parallel VSS blocks + gated merge
  4. DFFM: LayerNorm + Linear projection → F_Out

Reference: MBSSNet paper, Fig. 1 (right panel).
"""

import torch
import torch.nn as nn
from .vss_block import VSSBlock


class ChannelExchange(nn.Module):
    """
    Parameter-free channel exchange (from [13] Changer).
    Swaps the first half of channels between two inputs.
    """

    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio

    def forward(self, x1, x2):
        """
        x1, x2 : (B, C, H, W)
        Returns swapped x1, x2.
        """
        B, C, H, W = x1.shape
        n_swap = int(C * self.ratio)
        x1_out = torch.cat([x2[:, :n_swap], x1[:, n_swap:]], dim=1)
        x2_out = torch.cat([x1[:, :n_swap], x2[:, n_swap:]], dim=1)
        return x1_out, x2_out


class SFFM(nn.Module):
    """
    Shallow Feature Fusion Module.
    Two parallel VSS branches processing exchanged features,
    merged by element-wise addition.
    """

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.vss_s1 = VSSBlock(dim, d_state=d_state)
        self.vss_s2 = VSSBlock(dim, d_state=d_state)

    def forward(self, x_sar, x_opt):
        """
        x_sar, x_opt : (B, C, H, W) — after channel exchange
        Returns: f_s1, f_s2 : (B, C, H, W)
        """
        f_s1 = self.vss_s1(x_sar)
        f_s2 = self.vss_s2(x_opt)
        return f_s1, f_s2


class DFFM(nn.Module):
    """
    Deep Feature Fusion Module.
    Layer Norm + Linear → fused output.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        """x : (B, C, H, W)"""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()   # (B, H, W, C)
        x = self.linear(self.norm(x))
        x = x.permute(0, 3, 1, 2).contiguous()    # (B, C, H, W)
        return x


class CMFM(nn.Module):
    """
    Cross-Modal Fusion Module.

    Parameters
    ----------
    dim      : channel dimension at this scale
    d_state  : SSM state dimension
    exchange_ratio : fraction of channels to swap (default 0.5)
    """

    def __init__(self, dim, d_state=16, exchange_ratio=0.5):
        super().__init__()

        # 1. Layer norms
        self.norm_sar = nn.LayerNorm(dim)
        self.norm_opt = nn.LayerNorm(dim)

        # 2. Channel exchange
        self.channel_exchange = ChannelExchange(ratio=exchange_ratio)

        # 3. SFFM
        self.sffm = SFFM(dim, d_state=d_state)

        # 4. DFFM
        self.dffm = DFFM(dim)

    # ──────────────────────────────────────────────────────────
    def _apply_norm(self, x, norm):
        """Apply LayerNorm to (B, C, H, W) tensor."""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()    # (B, H, W, C)
        x = norm(x)
        return x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

    # ──────────────────────────────────────────────────────────
    def forward(self, f_sar, f_opt):
        """
        f_sar, f_opt : (B, C, H, W) — encoder features at one scale.
        Returns : f_out : (B, C, H, W) — fused feature map.
        """
        # 1. Normalise
        f_sar = self._apply_norm(f_sar, self.norm_sar)
        f_opt = self._apply_norm(f_opt, self.norm_opt)

        # 2. Channel exchange
        f_sar_ex, f_opt_ex = self.channel_exchange(f_sar, f_opt)

        # 3. SFFM  → two interactive feature maps
        f_s1, f_s2 = self.sffm(f_sar_ex, f_opt_ex)

        # Combine dual branches
        f_combined = f_s1 + f_s2

        # 4. DFFM  → final fused output
        f_out = self.dffm(f_combined)

        return f_out
