import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None

PATCH = 256          # patch side length (pixels)
TILE_SIZE = 5000
GRID_N = TILE_SIZE // PATCH   # 19 -> 361 patches per tile, non-overlapping

SEEN_CITIES = ["austin", "chicago", "tyrol-w", "vienna"]
HELD_OUT_CITY = "kitsap"

TRAIN_TILES = {city: [1, 2, 3] for city in SEEN_CITIES}   # 3 tiles/city -> train
VAL_TILES = {city: [4] for city in SEEN_CITIES}            # 1 tile/city -> in-domain val
TEST_TILES = {HELD_OUT_CITY: [1, 2]}                        # entire city -> out-of-domain test


def tile_list(split):
    if split == "train":
        d = TRAIN_TILES
    elif split == "val":
        d = VAL_TILES
    elif split == "test":
        d = TEST_TILES
    else:
        raise ValueError(split)
    return [(city, i) for city, idxs in d.items() for i in idxs]


class TileCache:
    """Loads whole tiles into memory once (uint8 arrays) and serves patch crops."""
    def __init__(self, data_dir, tiles):
        self.data_dir = data_dir
        self.images = {}
        self.masks = {}
        for city, i in tiles:
            name = f"{city}{i}"
            img = np.array(Image.open(f"{data_dir}/images/{name}.tif"))          # HxWx3 uint8
            mask = np.array(Image.open(f"{data_dir}/gt/{name}.tif"))             # HxW uint8 {0,255}
            self.images[name] = img
            self.masks[name] = (mask == 255).astype(np.uint8)                    # {0,1}

    def crop(self, name, row, col):
        r0, c0 = row * PATCH, col * PATCH
        img = self.images[name][r0:r0 + PATCH, c0:c0 + PATCH]
        mask = self.masks[name][r0:r0 + PATCH, c0:c0 + PATCH]
        return img, mask


class PatchDataset(Dataset):
    def __init__(self, cache: TileCache, tiles, augment=False):
        self.cache = cache
        self.augment = augment
        self.index = []
        for city, i in tiles:
            name = f"{city}{i}"
            for r in range(GRID_N):
                for c in range(GRID_N):
                    self.index.append((name, r, c))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        name, r, c = self.index[idx]
        img, mask = self.cache.crop(name, r, c)
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img, mask = img[::-1, :].copy(), mask[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()

        img_t = torch.from_numpy(img.transpose(2, 0, 1))   # 3xHxW
        mask_t = torch.from_numpy(mask).unsqueeze(0)        # 1xHxW
        return img_t, mask_t, name
