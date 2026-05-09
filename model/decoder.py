"""
Decoder for MBSSNet
====================
4-stage decoder with VSS blocks.
Each stage: Conv2d (channel adjust) → Upsample → VSS block.
Skip connections from CMFM outputs are added before each VSS block.

Final head: Conv2d(32, num_classes, 1x1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vss_block import VSSBlock


class PatchUpsample(nn.Module):
    """Upsample by 2× using ConvTranspose2d (learnable)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.up(x))


class DecoderStage(nn.Module):
    """
    Single decoder stage:
      skip + x → Conv1x1 → Upsample2x → VSSBlock
    """

    def __init__(self, in_ch, skip_ch, out_ch, d_state=16):
        super().__init__()
        self.channel_adj = nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = PatchUpsample(out_ch, out_ch)
        self.vss = VSSBlock(out_ch, d_state=d_state)

    def forward(self, x, skip=None):
        if skip is not None:
            # Ensure spatial dims match before concat
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn(self.channel_adj(x)))
        x = self.upsample(x)
        x = self.vss(x)
        return x


class Decoder(nn.Module):
    """
    MBSSNet 4-stage VSS decoder.

    Scales (for 256×256 input):
        stage 1:  (512) → 256   16×16  → 32×32
        stage 2:  (256+256) → 128   32×32  → 64×64
        stage 3:  (128+128) → 64    64×64  → 128×128
        stage 4:  (64+64) → 32     128×128 → 256×256

    Parameters
    ----------
    num_classes : number of output segmentation classes
    d_state     : SSM state dimension in VSS blocks
    """

    def __init__(self, num_classes=8, d_state=16):
        super().__init__()

        # Stage 1: 512 → 256 (no skip — deepest level)
        self.stage1 = DecoderStage(in_ch=512, skip_ch=0, out_ch=256, d_state=d_state)
        # Stage 2: 256 + 256 skip → 128
        self.stage2 = DecoderStage(in_ch=256, skip_ch=256, out_ch=128, d_state=d_state)
        # Stage 3: 128 + 128 skip → 64
        self.stage3 = DecoderStage(in_ch=128, skip_ch=128, out_ch=64, d_state=d_state)
        # Stage 4: 64 + 64 skip → 32
        self.stage4 = DecoderStage(in_ch=64, skip_ch=64, out_ch=32, d_state=d_state)

        # Classification head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, f_out1, f_out2, f_out3, f_out4):
        """
        Skip connections from CMFM outputs (shallow → deep):
            f_out1: (B, 64,  H/2,  W/2)
            f_out2: (B, 128, H/4,  W/4)
            f_out3: (B, 256, H/8,  W/8)
            f_out4: (B, 512, H/16, W/16)

        Returns: logits (B, num_classes, H, W)
        """
        # deepest → shallowest
        x = self.stage1(f_out4)              # (B,256, H/8,  W/8)
        x = self.stage2(x, f_out3)           # (B,128, H/4,  W/4)
        x = self.stage3(x, f_out2)           # (B, 64, H/2,  W/2)
        x = self.stage4(x, f_out1)           # (B, 32, H,    W)

        logits = self.head(x)                # (B, C, H, W)
        return logits
