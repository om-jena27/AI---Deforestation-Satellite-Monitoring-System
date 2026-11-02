import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv

class SiameseChangeNet(nn.Module):
    """
    Siamese Feature Difference Network for Satellite Change Detection (Siam-Diff).
    Inspired by TransSiamUNet (Ali et al., Nature Scientific Reports 2026).
    
    1. Twin weight-sharing encoder extracts hierarchical feature maps for T1 and T2.
    2. Explicit multi-scale feature subtraction |f(T1) - f(T2)| suppresses seasonal illumination jitter.
    3. Decoder reconstructs high-resolution deforestation disturbance boundaries.
    """
    def __init__(self, in_channels=4, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        curr_in = in_channels
        for f in features:
            self.encoder.append(DoubleConv(curr_in, f))
            curr_in = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.decoder = nn.ModuleList()
        for f in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, t1, t2):
        # Extract features for T1
        skips_t1 = []
        x1 = t1
        for block in self.encoder:
            x1 = block(x1)
            skips_t1.append(x1)
            x1 = self.pool(x1)
        b1 = self.bottleneck(x1)

        # Extract features for T2 (shared weights)
        skips_t2 = []
        x2 = t2
        for block in self.encoder:
            x2 = block(x2)
            skips_t2.append(x2)
            x2 = self.pool(x2)
        b2 = self.bottleneck(x2)

        # Bottleneck feature difference
        diff_b = torch.abs(b1 - b2)
        skips_t1 = skips_t1[::-1]
        skips_t2 = skips_t2[::-1]

        # Decoder with multi-scale feature difference fusion
        x = diff_b
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            diff_skip = torch.abs(skips_t1[idx // 2] - skips_t2[idx // 2])

            if x.shape != diff_skip.shape:
                x = F.interpolate(x, size=diff_skip.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat([diff_skip, x], dim=1)
            x = self.decoder[idx + 1](x)

        return torch.sigmoid(self.final_conv(x))
