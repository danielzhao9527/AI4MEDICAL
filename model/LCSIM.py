import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class CIU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CIU, self).__init__()

        self.branch_X = nn.Sequential(
            AttentionModule(in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.branch_Y = nn.Sequential(
            AttentionModule(in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1x1_X = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.AdaptiveAvgPool2d(1)  # AAP (Adaptive Avg Pool)
        )
        self.conv1x1_Y = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.AdaptiveAvgPool2d(1)  # AAP (Adaptive Avg Pool)
        )
        self.sigmoid_X = nn.Sigmoid()
        self.sigmoid_Y = nn.Sigmoid()

    def forward(self, X, Y):
        # First branch (X)
        X1 = self.branch_X(X)
        Y2 = self.conv1x1_Y(Y)
        Y2 = self.sigmoid_Y(Y2)
        X_out = X1 * Y2 + X1

        # Second branch (Y)
        Y1 = self.branch_Y(Y)
        X2 = self.conv1x1_X(X)
        X2 = self.sigmoid_X(X2)
        Y_out = Y1 * X2 + Y1

        return X_out, Y_out


class SIU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SIU, self).__init__()

        self.conv1x1_X = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv1x1_Y = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        kernel_size = 7
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, Y):
        X_pool = self.conv1x1_X(X)
        Y_pool = self.conv1x1_Y(Y)
        X_pool = torch.max(X_pool, dim=1, keepdim=True)[0]
        Y_pool = torch.max(Y_pool, dim=1, keepdim=True)[0]

        # Concatenation and sigmoid activation
        pool = torch.cat([X_pool, Y_pool], dim=1)
        pool = self.conv(pool)
        pool = self.sigmoid(pool)

        out = pool * X + pool * Y

        return out


class LCSIM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LCSIM, self).__init__()
        self.ciu = CIU(in_channels, out_channels)
        self.siu = SIU(in_channels, out_channels)

    def forward(self, X, Y):
        ciu_X, ciu_Y = self.ciu(X, Y)
        sit_out = self.siu(ciu_X, ciu_Y)

        return sit_out
