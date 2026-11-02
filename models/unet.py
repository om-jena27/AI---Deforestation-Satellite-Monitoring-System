import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ForestUNet(nn.Module):
    """
    Standard U-Net Architecture for Pixel-Level Forest Canopy Segmentation.
    Input: (B, in_channels, H, W) - RGB (3 channels) or RGB+NIR (4 channels).
    Output: (B, 1, H, W) - Forest probability map with Sigmoid activation.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return torch.sigmoid(self.final_conv(x))


class DeforestationUNet(ForestUNet):
    """
    Bi-temporal U-Net variant accepting concatenated T1 and T2 images (6 or 8 channels).
    Inherits all encoder-decoder logic from ForestUNet.
    """
    def __init__(self, in_channels=8, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__(in_channels=in_channels, out_channels=out_channels, features=features)


class CombinedBCEDiceLoss(nn.Module):
    """
    Combines Binary Cross-Entropy with Soft Dice Loss to handle class imbalance.
    """
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - ((2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth))
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.gap(x)
        x = self.conv(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18), out_channels=128):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP) for satellite canopy segmentation.
    Input: (B, 3, H, W)
    Output: (B, 1, H, W) - Sigmoid activation [0, 1]
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.aspp = ASPP(128, atrous_rates=(6, 12, 18), out_channels=128)
        self.low_project = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            DoubleConv(128 + 32, 64),
            nn.Conv2d(64, out_channels, 1),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        low = self.enc1(x)
        x = self.pool(low)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.enc3(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[2:], mode="bilinear", align_corners=False)
        low = self.low_project(low)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(x)

