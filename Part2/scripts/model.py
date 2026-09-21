import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2 -- the standard U-Net building block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Small U-Net for binary building segmentation.

    base=24, 4 downsampling stages -> channel widths [24, 48, 96, 192],
    bottleneck 384. ~4.4M trainable parameters (see param count printed at
    instantiation time in the training script) -- chosen to be trainable in a
    CPU-only sandbox within the assessment's time budget while keeping the
    full encoder-decoder-with-skips structure that gives U-Net its boundary-
    preservation advantage (see module docstring).
    """

    def __init__(self, in_ch=3, out_ch=1, base=24):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]

        self.enc1 = DoubleConv(in_ch, chs[0])
        self.enc2 = DoubleConv(chs[0], chs[1])
        self.enc3 = DoubleConv(chs[1], chs[2])
        self.enc4 = DoubleConv(chs[2], chs[3])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(chs[3], chs[3] * 2)

        self.up4 = nn.ConvTranspose2d(chs[3] * 2, chs[3], 2, stride=2)
        self.dec4 = DoubleConv(chs[3] * 2, chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = DoubleConv(chs[2] * 2, chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = DoubleConv(chs[1] * 2, chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = DoubleConv(chs[0] * 2, chs[0])

        self.out_conv = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)   # raw logits, shape (N, 1, H, W)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = UNet()
    n = count_params(m)
    x = torch.randn(2, 3, 256, 256)
    y = m(x)
    print(f"UNet params: {n:,}")
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}")
