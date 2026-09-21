import warnings
warnings.filterwarnings("ignore")

import os, time, json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_erosion


print(f"PyTorch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}  |  CPU count: {os.cpu_count()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
torch.set_num_threads(5)

CAT = {
        "blue": "#2a78d6", 
        "orange": "#eb6834", 
        "aqua": "#1baf7a", 
        "yellow": "#eda100",
        "magenta": "#e87ba4", 
        "green": "#008300", 
        "violet": "#4a3aa7", 
        "red": "#e34948"
    }
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK_PRIMARY,
    "text.color": INK_PRIMARY, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.grid.axis": "y",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
})

DATA_DIR = ".\\Part2\\data"          # expects data/images/*.tif and data/gt/*.tif alongside this notebook
CKPT_DIR = ".\\Part2\\checkpoints"

SEEN_CITIES = ["austin", "chicago", "tyrol-w", "vienna"]
HELD_OUT_CITY = "kitsap"
TILES_PER_SEEN_CITY = 4    # 3 train + 1 val, per city
HELD_OUT_TILES = 2         # test only — never seen in training or validation

all_tiles = [(c, i) for c in SEEN_CITIES for i in range(1, TILES_PER_SEEN_CITY + 1)]
all_tiles += [(HELD_OUT_CITY, i) for i in range(1, HELD_OUT_TILES + 1)]

records = []
for city, i in all_tiles:
    name = f"{city}{i}"
    mask = np.array(Image.open(f"{DATA_DIR}/gt/{name}.tif"))
    frac = (mask == 255).mean()
    records.append({"city": city, "tile": name, "building_frac": frac, "shape": mask.shape})
    print(f"{name:12s} shape={mask.shape}  building_pixel_fraction={frac:.4f}")

eda_df = pd.DataFrame(records)
print("\nPer-city mean building fraction:")
print(eda_df.groupby("city")["building_frac"].agg(["mean", "std", "min", "max"]).round(4))

overall_frac = eda_df["building_frac"].mean()
print(f"\nOverall mean building-pixel fraction: {overall_frac:.4f}  "
      f"(background:building ≈ {(1-overall_frac)/overall_frac:.2f} : 1 — confirms substantial class imbalance)")

city_colors = {"austin": CAT["blue"], "chicago": CAT["orange"], "tyrol-w": CAT["aqua"],
               "vienna": CAT["violet"], "kitsap": CAT["red"]}
fig, ax = plt.subplots(figsize=(11, 4))
colors = [city_colors[r["city"]] for r in records]
ax.bar([r["tile"] for r in records], [r["building_frac"]*100 for r in records], color=colors)
ax.axhline(overall_frac*100, color=INK_SECONDARY, lw=1.2, ls="--", label=f"Subset mean ({overall_frac*100:.1f}%)")
ax.set_ylabel("Building-pixel share (%)")
ax.set_title("Building-Pixel Coverage by Tile (color = city)")
ax.set_xticks(range(len(records)))
ax.set_xticklabels([r["tile"] for r in records], rotation=60, ha="right", fontsize=8)
handles = [Patch(color=c, label=city) for city, c in city_colors.items()]
ax.legend(handles=handles + [plt.Line2D([0],[0], color=INK_SECONDARY, lw=1.2, ls="--", label="subset mean")],
          loc="upper right", ncol=2, fontsize=8)
fig.tight_layout(); plt.show()

sample_tiles = [("austin", 1), ("chicago", 1), ("tyrol-w", 1), ("vienna", 1), ("kitsap", 1)]
fig, axes = plt.subplots(2, len(sample_tiles), figsize=(3.1*len(sample_tiles), 6.4))
for col, (city, i) in enumerate(sample_tiles):
    name = f"{city}{i}"
    img = np.array(Image.open(f"{DATA_DIR}/images/{name}.tif"))
    mask = np.array(Image.open(f"{DATA_DIR}/gt/{name}.tif"))
    img_small, mask_small = img[::8, ::8], mask[::8, ::8]
    axes[0, col].imshow(img_small); axes[0, col].set_title(name, fontsize=10); axes[0, col].axis("off")
    axes[1, col].imshow(img_small)
    overlay = np.zeros((*mask_small.shape, 4))
    overlay[mask_small == 255] = [0.93, 0.29, 0.18, 0.55]
    axes[1, col].imshow(overlay); axes[1, col].axis("off")
axes[0, 0].set_ylabel("Aerial image", fontsize=10)
axes[1, 0].set_ylabel("+ building mask", fontsize=10)
fig.suptitle("Sample Tile per City: Aerial Image (top) and Ground-Truth Building Mask (bottom)",
             y=1.02, fontweight="bold", fontsize=12)
fig.tight_layout(); plt.show()


# Data Partitioning
PATCH = 256
TILE_SIZE = 5000
GRID_N = TILE_SIZE // PATCH   # 19 -> 361 non-overlapping patches per tile

TRAIN_TILES = {city: [1, 2, 3] for city in SEEN_CITIES}
VAL_TILES = {city: [4] for city in SEEN_CITIES}
TEST_TILES = {HELD_OUT_CITY: [1, 2]}

def tile_list(split):
    d = {"train": TRAIN_TILES, "val": VAL_TILES, "test": TEST_TILES}[split]
    return [(city, i) for city, idxs in d.items() for i in idxs]

class TileCache:
    '''Loads whole tiles into memory once and serves non-overlapping patch crops.'''
    def __init__(self, data_dir, tiles):
        self.images, self.masks = {}, {}
        for city, i in tiles:
            name = f"{city}{i}"
            self.images[name] = np.array(Image.open(f"{data_dir}/images/{name}.tif"))
            self.masks[name] = (np.array(Image.open(f"{data_dir}/gt/{name}.tif")) == 255).astype(np.uint8)

    def crop(self, name, row, col):
        r0, c0 = row * PATCH, col * PATCH
        return (self.images[name][r0:r0+PATCH, c0:c0+PATCH],
                self.masks[name][r0:r0+PATCH, c0:c0+PATCH])

class PatchDataset(Dataset):
    def __init__(self, cache, tiles, augment=False):
        self.cache, self.augment = cache, augment
        self.index = [(f"{city}{i}", r, c) for city, i in tiles
                      for r in range(GRID_N) for c in range(GRID_N)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        name, r, c = self.index[idx]
        img, mask = self.cache.crop(name, r, c)
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5: img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if np.random.rand() < 0.5: img, mask = img[::-1, :].copy(), mask[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k: img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()
        return (torch.from_numpy(img.transpose(2, 0, 1)),
                torch.from_numpy(mask).unsqueeze(0), name)

print(f"PATCH={PATCH}, non-overlapping patches/tile = {GRID_N}x{GRID_N} = {GRID_N*GRID_N}")
for split in ["train", "val", "test"]:
    tiles = tile_list(split)
    print(f"{split:5s}: tiles={tiles}  -> {len(tiles)*GRID_N*GRID_N} patches")

# --- Verify the partition end-to-end and check patch-level class balance per split ---
t0 = time.time()
train_cache = TileCache(DATA_DIR, tile_list("train"))
val_cache = TileCache(DATA_DIR, tile_list("val"))
test_cache = TileCache(DATA_DIR, tile_list("test"))
print(f"Tile caches loaded in {time.time()-t0:.1f}s")

train_ds = PatchDataset(train_cache, tile_list("train"), augment=True)
val_ds = PatchDataset(val_cache, tile_list("val"), augment=False)
test_ds = PatchDataset(test_cache, tile_list("test"), augment=False)

for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
    rng = np.random.RandomState(0)
    idxs = rng.choice(len(ds), size=min(200, len(ds)), replace=False)
    fracs = np.array([ds[i][1].mean().item() for i in idxs])
    print(f"{name:5s} n={len(ds):5d}  patch building-fraction: mean={fracs.mean():.4f} "
          f"empty-patches={((fracs==0).mean()*100):.1f}%")


################# MODEL ARCHITECTURE ########################
print("\n MODEL ARCHITECTURE")

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=16):
        super().__init__()
        chs = [base, base*2, base*4, base*8]
        self.enc1, self.enc2, self.enc3, self.enc4 = (DoubleConv(in_ch, chs[0]), DoubleConv(chs[0], chs[1]),
                                                        DoubleConv(chs[1], chs[2]), DoubleConv(chs[2], chs[3]))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[3], chs[3]*2)
        self.up4 = nn.ConvTranspose2d(chs[3]*2, chs[3], 2, stride=2); self.dec4 = DoubleConv(chs[3]*2, chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2);   self.dec3 = DoubleConv(chs[2]*2, chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2);   self.dec2 = DoubleConv(chs[1]*2, chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2);   self.dec1 = DoubleConv(chs[0]*2, chs[0])
        self.out_conv = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

model = UNet(base=16)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
x_test = torch.randn(2, 3, 256, 256)
y_test = model(x_test)
print(f"\nUNet(base=16) parameters: {n_params:,}")
print(f"sanity check: input {tuple(x_test.shape)} -> output {tuple(y_test.shape)}")


################# LOSS FUNCTION ########################
def boundary_band(mask, iterations=2):
    '''mask: (N,1,H,W) float {0,1}. dilation-minus-erosion via repeated 3x3 max/min pooling.'''
    dil = ero = mask
    for _ in range(iterations):
        dil = F.max_pool2d(dil, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-ero, kernel_size=3, stride=1, padding=1)
    return (dil - ero).clamp(0, 1)

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth = smooth
    def forward(self, logits, target):
        prob = torch.sigmoid(logits).reshape(logits.size(0), -1)
        target = target.reshape(target.size(0), -1)
        inter = (prob * target).sum(dim=1)
        union = prob.sum(dim=1) + target.sum(dim=1)
        return 1 - ((2*inter + self.smooth) / (union + self.smooth)).mean()

class BoundaryWeightedBCE(nn.Module):
    def __init__(self, pos_weight=1.0, boundary_weight=5.0, boundary_iters=2):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.boundary_weight, self.boundary_iters = boundary_weight, boundary_iters
    def forward(self, logits, target):
        per_pixel = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight, reduction="none")
        with torch.no_grad():
            band = boundary_band(target, self.boundary_iters)
            weight_map = 1.0 + (self.boundary_weight - 1.0) * band
        return (per_pixel * weight_map).mean()

class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=1.0, boundary_weight=5.0, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = BoundaryWeightedBCE(pos_weight, boundary_weight)
        self.dice = SoftDiceLoss()
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
    def forward(self, logits, target):
        bce_l, dice_l = self.bce(logits, target), self.dice(logits, target)
        total = self.bce_weight*bce_l + self.dice_weight*dice_l
        return total, {"bce": bce_l.item(), "dice": dice_l.item(), "total": total.item()}

# exact pos_weight from the full train-split pixel counts
total_pix = sum(m.size for m in train_cache.masks.values())
pos_pix = sum(int(m.sum()) for m in train_cache.masks.values())
pos_weight = (total_pix - pos_pix) / pos_pix
print(f"\ntrain building-pixel fraction: {pos_pix/total_pix:.4f}  ->  pos_weight = {pos_weight:.3f}")

loss_fn = CombinedLoss(pos_weight=pos_weight, boundary_weight=5.0).to(device)

# --- Visualize the boundary band on a real mask patch, to make the loss's boundary term concrete ---
sample_img, sample_mask, sample_name = train_ds[np.random.RandomState(3).randint(len(train_ds))]
band = boundary_band(sample_mask.unsqueeze(0), iterations=2)[0, 0].numpy()

fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
axes[0].imshow(sample_img.numpy().transpose(1, 2, 0)); axes[0].set_title(f"Aerial patch ({sample_name})"); axes[0].axis("off")
axes[1].imshow(sample_mask[0].numpy(), cmap="gray"); axes[1].set_title("Ground-truth mask"); axes[1].axis("off")
axes[2].imshow(band, cmap="magma"); axes[2].set_title("Boundary weight band (loss term 3)"); axes[2].axis("off")
fig.tight_layout(); plt.show()

EPS = 1e-7

def _np(x): return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x).astype(np.uint8)

def iou_score(pred, target):
    pred, target = _np(pred), _np(target)
    union = np.logical_or(pred, target).sum()
    return 1.0 if union == 0 else np.logical_and(pred, target).sum() / (union + EPS)

def dice_score(pred, target):
    pred, target = _np(pred), _np(target)
    denom = pred.sum() + target.sum()
    return 1.0 if denom == 0 else 2*np.logical_and(pred, target).sum() / (denom + EPS)

def precision_score(pred, target):
    pred, target = _np(pred), _np(target)
    tp = np.logical_and(pred, target).sum(); fp = np.logical_and(pred, np.logical_not(target)).sum()
    if tp+fp == 0: return 1.0 if target.sum() == 0 else 0.0
    return tp / (tp+fp+EPS)

def recall_score(pred, target):
    pred, target = _np(pred), _np(target)
    tp = np.logical_and(pred, target).sum(); fn = np.logical_and(np.logical_not(pred), target).sum()
    if tp+fn == 0: return 1.0 if pred.sum() == 0 else 0.0
    return tp / (tp+fn+EPS)

def _boundary_mask(mask, dilation_ratio=0.02):
    h, w = mask.shape
    radius = max(1, round(dilation_ratio * (h+w) / 2))
    eroded = binary_erosion(mask.astype(bool), iterations=radius, border_value=0)
    return np.logical_and(mask.astype(bool), np.logical_not(eroded))

def boundary_iou(pred, target, dilation_ratio=0.02):
    pred, target = _np(pred), _np(target)
    pb, tb = _boundary_mask(pred, dilation_ratio), _boundary_mask(target, dilation_ratio)
    union = np.logical_or(pb, tb).sum()
    return 1.0 if union == 0 else np.logical_and(pb, tb).sum() / (union + EPS)

def compute_all(pred, target):
    return {"iou": iou_score(pred, target), "dice": dice_score(pred, target),
            "precision": precision_score(pred, target), "recall": recall_score(pred, target),
            "boundary_iou": boundary_iou(pred, target)}

print("Sanity check on a synthetic mask pair:",
      compute_all((np.random.RandomState(0).rand(64,64)>0.8).astype(np.uint8),
                  (np.random.RandomState(0).rand(64,64)>0.8).astype(np.uint8)))



hist_path = "./Part2/train_history_gpu.csv"
if os.path.exists(hist_path):
    hist = pd.read_csv(hist_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(hist["epoch"], hist["train_loss"], color=CAT["blue"], lw=2.2, marker="o")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss (combined)"); axes[0].set_title("Full Training Run: Loss")
    axes[1].plot(hist["epoch"], hist["val_iou_quick"], color=CAT["aqua"], lw=2.2, marker="o", label="Quick-val IoU")
    axes[1].plot(hist["epoch"], hist["val_dice_quick"], color=CAT["orange"], lw=2.2, marker="o", label="Quick-val Dice")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_title("Full Training Run: Monitoring Metrics")
    axes[1].legend(loc="lower right")
    fig.tight_layout(); plt.show()
    print(hist.round(4).to_string(index=False))
else:
    print("train_history_gpu.csv not found alongside this notebook -- run scripts/train_gpu.py first.")


def evaluate_split(model, ds, split_name):
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model.eval()
    rows = []
    with torch.no_grad():
        for xb, yb, names in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = (torch.sigmoid(model(xb)) > 0.5).float()
            for p, t, name in zip(pred, yb, names):
                p_np, t_np = p[0].detach().cpu().numpy().astype(np.uint8), t[0].detach().cpu().numpy().astype(np.uint8)
                row = compute_all(p_np, t_np)
                row["name"] = name; row["gt_frac"] = float(t_np.mean()); row["pred_frac"] = float(p_np.mean())
                rows.append(row)
    keys = ["iou", "dice", "precision", "recall", "boundary_iou"]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows])) for k in keys}
    print(f"\n=== {split_name} ({len(rows)} patches) ===")
    print("Whole-split (all patches, incl. empty):")
    for k in keys:
        print(f"  {k:14s} mean={agg[k]:.4f}  std={std[k]:.4f}")

    nonempty = [r for r in rows if r["gt_frac"] > 0]
    empty = [r for r in rows if r["gt_frac"] == 0]
    print(f"\npatches with >=1 building pixel: {len(nonempty)}/{len(rows)} ({len(nonempty)/len(rows)*100:.1f}%)")
    nonempty_agg = {}
    if nonempty:
        nonempty_agg = {k: float(np.mean([r[k] for r in nonempty])) for k in keys}
        print("Building-containing patches only (the metric that isn't inflated by trivial empty-vs-empty matches):")
        for k in keys:
            print(f"  {k:14s} mean={nonempty_agg[k]:.4f}")
    if empty:
        fp_rate = float(np.mean([r["pred_frac"] > 0 for r in empty]))
        print(f"\nFalse-positive rate on truly-empty patches (model predicts a building where there is none): {fp_rate*100:.1f}%")

    return rows, agg, std, nonempty_agg

eval_model = UNet(base=16).to(device)
ckpt_path = os.path.join(CKPT_DIR, "best_unet.pt")
if os.path.exists(ckpt_path):
    eval_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded fully-trained checkpoint: {ckpt_path} on {device}")
else:
    print("Full checkpoint not found -- run scripts/train_gpu.py first.")

val_rows, val_agg, val_std, val_ne = evaluate_split(eval_model, val_ds, "VAL (in-domain, held-out tiles)")
test_rows, test_agg, test_std, test_ne = evaluate_split(eval_model, test_ds, "TEST (out-of-domain, Kitsap, held-out city)")

print("\n=== Naive whole-split gap (VAL - TEST), all patches ===")
for k in ["iou", "dice", "precision", "recall", "boundary_iou"]:
    print(f"  {k:14s} gap = {val_agg[k]-test_agg[k]:+.4f}")
print("\n=== Real gap on building-containing patches only (VAL - TEST) ===")
for k in ["iou", "dice", "precision", "recall", "boundary_iou"]:
    print(f"  {k:14s} gap = {val_ne[k]-test_ne[k]:+.4f}")


def qualitative_panel(model, ds, split_name, n_examples=3):
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model.eval()
    scored = []
    with torch.no_grad():
        for xb, yb, names in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            probs = torch.sigmoid(model(xb)); pred = (probs > 0.5).float()
            for i in range(xb.size(0)):
                img = xb[i].detach().cpu().numpy().transpose(1, 2, 0)
                t_np = yb[i, 0].detach().cpu().numpy().astype(np.uint8)
                p_np = pred[i, 0].detach().cpu().numpy().astype(np.uint8)
                scored.append({"img": img, "gt": t_np, "pred": p_np,
                                "iou": iou_score(p_np, t_np), "name": names[i], "gt_frac": t_np.mean()})
    wb = [s for s in scored if s["gt_frac"] > 0.01]
    wb.sort(key=lambda s: -s["iou"])
    picks = wb[:n_examples] + wb[-n_examples:]
    labels = [f"best #{i+1} (IoU={s['iou']:.2f})" for i, s in enumerate(wb[:n_examples])] + \
             [f"failure #{i+1} (IoU={s['iou']:.2f})" for i, s in enumerate(wb[-n_examples:])]

    fig, axes = plt.subplots(3, len(picks), figsize=(2.6*len(picks), 8.2))
    for col, (s, label) in enumerate(zip(picks, labels)):
        axes[0, col].imshow(s["img"]); axes[0, col].set_title(f"{s['name']}\n{label}", fontsize=8.5); axes[0, col].axis("off")
        axes[1, col].imshow(s["gt"], cmap="gray", vmin=0, vmax=1); axes[1, col].axis("off")
        axes[2, col].imshow(s["pred"], cmap="gray", vmin=0, vmax=1); axes[2, col].axis("off")
    axes[0, 0].set_ylabel("Aerial patch", fontsize=10)
    axes[1, 0].set_ylabel("Ground truth", fontsize=10)
    axes[2, 0].set_ylabel("Prediction", fontsize=10)
    fig.suptitle(f"Qualitative Assessment -- {split_name}: best extractions (left) vs. failure cases (right)",
                 y=1.02, fontweight="bold", fontsize=12)
    fig.tight_layout(); plt.show()

qualitative_panel(eval_model, val_ds, "In-Domain Validation")
qualitative_panel(eval_model, test_ds, "Out-of-Domain Test (Kitsap)")