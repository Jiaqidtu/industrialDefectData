"""ADMSB: Adaptive Deformable Multi-Scale Bottleneck.

Drop-in replacement for the standard Bottleneck inside a C2f block.
Designed for NEU-DET style surface defects, where the targets range from
thin elongated cracks to round pitted surfaces and large patches, so a
fixed 3x3 geometry is a poor fit.

Three parallel branches see the same reduced feature map:
  A  deformable 3x3  -> learned sampling grid, follows irregular edges
  B  depthwise-sep 5x5 -> large receptive field for patches / pitted areas
  C  avgpool 3x3     -> smooth background prior, damps the high-frequency
                        noise the deformable branch can introduce
A lightweight SE head then re-weights the concatenated 1.5C channels, so the
network learns *which* geometry to trust per image instead of just summing
the branches.

Only external dependency beyond torch is torchvision.ops.deform_conv2d.
"""

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

from .layers import Bottleneck, Conv


class DeformBranch(nn.Module):
    """Branch A: 3x3 deformable convolution with a learned offset field.

    The offset predictor is a separate 1x1 conv emitting 2 * 3 * 3 = 18
    channels (dy, dx per kernel tap). It is zero-initialised so the module
    starts out identical to a plain 3x3 conv and only gradually deforms,
    which keeps early training stable.
    """

    def __init__(self, c, k=3):
        super().__init__()
        self.k = k
        self.offset = nn.Conv2d(c, 2 * k * k, kernel_size=1, stride=1,
                                padding=0, bias=True)
        # start from the regular sampling grid
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

        self.weight = nn.Parameter(torch.empty(c, c, k, k))
        nn.init.kaiming_normal_(self.weight, mode='fan_out',
                                nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(c))
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # x: [B, C/2, H, W]
        off = self.offset(x)                                # [B, 18, H, W]
        y = deform_conv2d(x, off, self.weight, self.bias,
                          stride=1, padding=self.k // 2)     # [B, C/2, H, W]
        return self.act(self.bn(y))                          # [B, C/2, H, W]


class DWSep5x5(nn.Module):
    """Branch B: 5x5 depthwise + 1x1 pointwise (big context, few params)."""

    def __init__(self, c):
        super().__init__()
        self.dw = Conv(c, c, k=5, s=1, g=c)   # depthwise 5x5
        self.pw = Conv(c, c, k=1, s=1)        # pointwise 1x1

    def forward(self, x):
        # x: [B, C/2, H, W]
        x = self.dw(x)   # [B, C/2, H, W]
        x = self.pw(x)   # [B, C/2, H, W]
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation over the concatenated branch features."""

    def __init__(self, c, r=8):
        super().__init__()
        c_hidden = max(4, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c, c_hidden, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(c_hidden, c, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)      # [B, 1.5C]
        s = self.relu(self.fc1(s))       # [B, 1.5C/r]
        s = self.gate(self.fc2(s))       # [B, 1.5C]
        return x * s.view(b, c, 1, 1)    # [B, 1.5C, H, W]


class ADMSB(nn.Module):
    """Adaptive Deformable Multi-Scale Bottleneck.

    Args:
        c1: input channels
        c2: output channels (must equal c1 for the residual path)
        e:  reduction ratio for the internal width (0.5 -> C/2)
        r:  SE reduction ratio
        shortcut: keep the residual connection
    """

    def __init__(self, c1, c2=None, shortcut=True, e=0.5, r=8):
        super().__init__()
        c2 = c1 if c2 is None else c2
        c_ = int(c1 * e)                 # C/2
        self.c_ = c_
        self.add = shortcut and c1 == c2

        self.reduce = Conv(c1, c_, k=1, s=1)                # 1x1 down
        self.branch_a = DeformBranch(c_, k=3)               # deformable
        self.branch_b = DWSep5x5(c_)                        # 5x5 dw-sep
        self.branch_c = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.se = SE(3 * c_, r=r)                           # 1.5C gate
        self.expand = Conv(3 * c_, c2, k=1, s=1)            # 1x1 up

    def forward(self, x):
        # x:                                        [B, C,     H, W]
        y = self.reduce(x)                         # [B, C/2,   H, W]

        a = self.branch_a(y)                       # [B, C/2,   H, W]
        b = self.branch_b(y)                       # [B, C/2,   H, W]
        c = self.branch_c(y)                       # [B, C/2,   H, W]

        m = torch.cat((a, b, c), dim=1)            # [B, 3C/2,  H, W]
        m = self.se(m)                             # [B, 3C/2,  H, W]  weighted
        out = self.expand(m)                       # [B, C,     H, W]

        return x + out if self.add else out        # [B, C,     H, W]


class C2f_ADMSB(nn.Module):
    """C2f whose first `n_admsb` bottlenecks are ADMSB blocks.

    Used for the P3 / P4 stages only; P5 keeps the stock C2f so the extra
    cost stays where the small and mid-size defects are.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, n_admsb=2):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        blocks = []
        for i in range(n):
            if i < n_admsb:
                blocks.append(ADMSB(self.c, self.c, shortcut=True, e=0.5))
            else:
                blocks.append(Bottleneck(self.c, self.c, shortcut, e=1.0,
                                         k=(3, 3)))
        self.m = nn.ModuleList(blocks)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))          # 2 x [B, c, H, W]
        for m in self.m:
            y.append(m(y[-1]))                     # [B, c, H, W]
        return self.cv2(torch.cat(y, 1))           # [B, c2, H, W]
