import numpy as np
from scipy.ndimage import binary_erosion

EPS = 1e-7


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(np.uint8)


def iou_score(pred, target):
    pred, target = _to_numpy(pred), _to_numpy(target)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0   # both empty -> perfect agreement
    return inter / (union + EPS)


def dice_score(pred, target):
    pred, target = _to_numpy(pred), _to_numpy(target)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2 * inter / (denom + EPS)


def precision_score(pred, target):
    pred, target = _to_numpy(pred), _to_numpy(target)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    if tp + fp == 0:
        return 1.0 if target.sum() == 0 else 0.0
    return tp / (tp + fp + EPS)


def recall_score(pred, target):
    pred, target = _to_numpy(pred), _to_numpy(target)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()
    if tp + fn == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return tp / (tp + fn + EPS)


def _boundary_mask(mask, dilation_ratio=0.02):
    """Cheng et al. (2021)-style boundary extraction: erode the mask by a
    radius proportional to image size, subtract from the original to get a
    thin boundary band. dilation_ratio=0.02 on a 256px patch -> ~5px band.
    """
    h, w = mask.shape
    radius = max(1, round(dilation_ratio * (h + w) / 2))
    eroded = binary_erosion(mask.astype(bool), iterations=radius, border_value=0)
    return np.logical_and(mask.astype(bool), np.logical_not(eroded))


def boundary_iou(pred, target, dilation_ratio=0.02):
    pred, target = _to_numpy(pred), _to_numpy(target)
    pred_b = _boundary_mask(pred, dilation_ratio)
    target_b = _boundary_mask(target, dilation_ratio)
    inter = np.logical_and(pred_b, target_b).sum()
    union = np.logical_or(pred_b, target_b).sum()
    if union == 0:
        return 1.0
    return inter / (union + EPS)


def compute_all(pred, target, dilation_ratio=0.02):
    """pred, target: 2D {0,1} arrays for a single image/patch."""
    return {
        "iou": iou_score(pred, target),
        "dice": dice_score(pred, target),
        "precision": precision_score(pred, target),
        "recall": recall_score(pred, target),
        "boundary_iou": boundary_iou(pred, target, dilation_ratio),
    }


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    target = (rng.rand(64, 64) > 0.8).astype(np.uint8)
    pred = target.copy()
    pred[0:5, 0:5] = 1 - pred[0:5, 0:5]   # inject some errors
    print(compute_all(pred, target))
