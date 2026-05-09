"""
Modified ResNet-34 Backbone for MBSSNet PSNet
==============================================
Key modification from the paper:
  - First conv layer uses stride=1 (instead of stride=2) so that the
    feature map after layer1 is (H/2, W/2) rather than (H/4, W/4),
    preserving more spatial detail.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (two 3x3 convs with residual)."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ModifiedResNet34(nn.Module):
    """
    ResNet-34 with stride-1 first conv for larger feature maps.

    Output feature scales (for 256×256 input):
        F1: (B, 64,  128, 128)   — after layer1
        F2: (B, 128,  64,  64)   — after layer2
        F3: (B, 256,  32,  32)   — after layer3
        F4: (B, 512,  16,  16)   — after layer4

    Parameters
    ----------
    in_channels : number of input channels (4 for optical, 1 for SAR)
    pretrained  : load ImageNet weights where possible
    """

    layers_cfg = [3, 4, 6, 3]   # ResNet-34 block counts

    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()

        # ── stem (modified: stride=1) ──
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── residual layers ──
        self.layer1 = self._make_layer(64,  64,  self.layers_cfg[0], stride=1)
        self.layer2 = self._make_layer(64,  128, self.layers_cfg[1], stride=2)
        self.layer3 = self._make_layer(128, 256, self.layers_cfg[2], stride=2)
        self.layer4 = self._make_layer(256, 512, self.layers_cfg[3], stride=2)

        self._init_weights(in_channels, pretrained)

    # ──────────────────────────────────────────────────────────
    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    # ──────────────────────────────────────────────────────────
    def _init_weights(self, in_channels, pretrained):
        """Load pretrained weights, adapting the first conv if channels differ."""
        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                           nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            return

        # Load standard ResNet-34 weights
        pretrained_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        state = pretrained_model.state_dict()

        # Adapt conv1 weights for non-3-channel input
        conv1_key = 'conv1.weight'
        pretrained_conv1 = state[conv1_key]            # (64, 3, 7, 7)
        if in_channels != 3:
            if in_channels == 1:
                # Average RGB weights
                new_w = pretrained_conv1.mean(dim=1, keepdim=True)
            elif in_channels == 4:
                # Copy RGB + init NIR from mean
                nir_w = pretrained_conv1.mean(dim=1, keepdim=True)
                new_w = torch.cat([pretrained_conv1, nir_w], dim=1)
            else:
                # General: repeat / truncate
                reps = (in_channels // 3) + 1
                new_w = pretrained_conv1.repeat(1, reps, 1, 1)[:, :in_channels]
            state[conv1_key] = new_w

        # Load layer weights (skip fc, avgpool — we don't use them)
        own_state = self.state_dict()
        for name, param in state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)

    # ──────────────────────────────────────────────────────────
    def forward(self, x):
        """
        Returns 4 multi-scale feature maps.
        """
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 64, H, W)  stride-1
        x = self.maxpool(x)                       # (B, 64, H/2, W/2)

        f1 = self.layer1(x)                       # (B,  64, H/2, W/2)
        f2 = self.layer2(f1)                      # (B, 128, H/4, W/4)
        f3 = self.layer3(f2)                      # (B, 256, H/8, W/8)
        f4 = self.layer4(f3)                      # (B, 512, H/16, W/16)

        return f1, f2, f3, f4
