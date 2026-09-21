import torch
import torch.nn as nn
import torch.nn.functional as F


def _boundary_band(mask, iterations=2):
    """mask: (N,1,H,W) float {0,1}. Returns a {0,1} band around mask edges via
    (dilation - erosion), computed with 3x3 max/min pooling (erosion = -maxpool(-x)).
    `iterations` widens the band by that many 3x3 passes (2 -> ~2px each side).
    """
    dil = mask
    ero = mask
    for _ in range(iterations):
        dil = F.max_pool2d(dil, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-ero, kernel_size=3, stride=1, padding=1)
    return (dil - ero).clamp(0, 1)   # 1 on the boundary band, 0 elsewhere


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        prob = prob.reshape(prob.size(0), -1)
        target = target.reshape(target.size(0), -1)
        intersection = (prob * target).sum(dim=1)
        union = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BoundaryWeightedBCE(nn.Module):
    def __init__(self, pos_weight=1.0, boundary_weight=5.0, boundary_iters=2):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.boundary_weight = boundary_weight
        self.boundary_iters = boundary_iters

    def forward(self, logits, target):
        per_pixel = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        with torch.no_grad():
            band = _boundary_band(target, iterations=self.boundary_iters)
            weight_map = 1.0 + (self.boundary_weight - 1.0) * band
        return (per_pixel * weight_map).mean()


class CombinedLoss(nn.Module):
    """bce_weight * BoundaryWeightedBCE(pos_weight) + dice_weight * SoftDiceLoss"""

    def __init__(self, pos_weight=1.0, boundary_weight=5.0, boundary_iters=2,
                 bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = BoundaryWeightedBCE(pos_weight, boundary_weight, boundary_iters)
        self.dice = SoftDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        bce_l = self.bce(logits, target)
        dice_l = self.dice(logits, target)
        total = self.bce_weight * bce_l + self.dice_weight * dice_l
        return total, {"bce": bce_l.item(), "dice": dice_l.item(), "total": total.item()}


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.85).float()
    loss_fn = CombinedLoss(pos_weight=5.3, boundary_weight=5.0)
    total, parts = loss_fn(logits, target)
    print("total:", total.item(), "parts:", parts)
