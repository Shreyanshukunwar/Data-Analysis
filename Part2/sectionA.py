import os, time, json
import torch
import matplotlib as mpl
import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import Dataset, DataLoader


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

DATA_DIR = "./Part2/data"          # expects data/images/*.tif and data/gt/*.tif alongside this notebook
CKPT_DIR = "./Part2/checkpoints"

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
