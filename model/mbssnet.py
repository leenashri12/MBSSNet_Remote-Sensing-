"""
MBSSNet — Mamba-Based Joint Semantic Segmentation Network
==========================================================
Encoder–decoder architecture for fusing Optical and SAR images:
  Encoder:  dual-branch modified ResNet-34 (PSNet) + CMFM at each scale
  Decoder:  4-stage VSS decoder with multi-scale skip connections

Reference: Li et al., "MBSSNet: A Mamba-Based Joint Semantic
Segmentation Network for Optical and SAR Images", IEEE, 2025.
"""

import torch
import torch.nn as nn
from .resnet34 import ModifiedResNet34
from .cmfm import CMFM
from .decoder import Decoder


class MBSSNet(nn.Module):
    """
    MBSSNet end-to-end segmentation network.

    Parameters
    ----------
    opt_channels : input channels for optical branch (default 4 for GF-1)
    sar_channels : input channels for SAR branch (default 1)
    num_classes  : number of segmentation classes (default 8 for WHU)
    pretrained   : use ImageNet-pretrained ResNet-34 weights
    d_state      : SSM state dimension for SS2D / VSS blocks
    """

    def __init__(self, opt_channels=4, sar_channels=1, num_classes=8,
                 pretrained=True, d_state=16):
        super().__init__()

        # ── Encoder: Pseudo-Siamese Network (PSNet) ──────────
        self.encoder_opt = ModifiedResNet34(in_channels=opt_channels,
                                            pretrained=pretrained)
        self.encoder_sar = ModifiedResNet34(in_channels=sar_channels,
                                            pretrained=pretrained)

        # ── CMFM at each of the 4 scales ────────────────────
        self.cmfm1 = CMFM(dim=64,  d_state=d_state)
        self.cmfm2 = CMFM(dim=128, d_state=d_state)
        self.cmfm3 = CMFM(dim=256, d_state=d_state)
        self.cmfm4 = CMFM(dim=512, d_state=d_state)

        # ── Decoder ──────────────────────────────────────────
        self.decoder = Decoder(num_classes=num_classes, d_state=d_state)

    # ──────────────────────────────────────────────────────────
    def forward(self, optical, sar):
        """
        optical : (B, opt_channels, H, W)
        sar     : (B, sar_channels, H, W)

        Returns : logits (B, num_classes, H, W)
        """
        # 1. Dual-branch encoding  (Eq. 1)
        f1_opt, f2_opt, f3_opt, f4_opt = self.encoder_opt(optical)
        f1_sar, f2_sar, f3_sar, f4_sar = self.encoder_sar(sar)

        # 2. Cross-modal fusion at each scale  (Eq. 2)
        f_out1 = self.cmfm1(f1_sar, f1_opt)    # (B,  64, H/2, W/2)
        f_out2 = self.cmfm2(f2_sar, f2_opt)    # (B, 128, H/4, W/4)
        f_out3 = self.cmfm3(f3_sar, f3_opt)    # (B, 256, H/8, W/8)
        f_out4 = self.cmfm4(f4_sar, f4_opt)    # (B, 512, H/16,W/16)

        # 3. Decode  (Eq. 3)
        logits = self.decoder(f_out1, f_out2, f_out3, f_out4)

        return logits
