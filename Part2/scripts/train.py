import sys, os, time, json, csv
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TileCache, PatchDataset, tile_list
from model import UNet, count_params
from losses import CombinedLoss
import metrics as M

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

BASE_CHANNELS = 16
BATCH = 16
STEPS_PER_EPOCH = 120
EPOCHS = 8
VAL_QUICK_N = 240
LR = 1e-3
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(os.cpu_count())


def log_line(f, msg):
    print(msg, flush=True)
    f.write(msg + "\n")
    f.flush()


def quick_eval(model, loader_iter_fn, n_batches, device="cpu"):
    model.eval()
    ious, dices = [], []
    with torch.no_grad():
        for xb, yb, _ in loader_iter_fn(n_batches):
            logits = model(xb)
            pred = (torch.sigmoid(logits) > 0.5).float()
            for p, t in zip(pred, yb):
                p_np = p[0].numpy().astype(np.uint8)
                t_np = t[0].numpy().astype(np.uint8)
                ious.append(M.iou_score(p_np, t_np))
                dices.append(M.dice_score(p_np, t_np))
    model.train()
    return float(np.mean(ious)), float(np.mean(dices))


def main():
    log_path = os.path.join(os.path.dirname(__file__), "..", "train_log.txt")
    logf = open(log_path, "w")

    log_line(logf, f"CPU count: {os.cpu_count()}")
    log_line(logf, f"Config: base={BASE_CHANNELS} batch={BATCH} steps/epoch={STEPS_PER_EPOCH} "
                    f"epochs={EPOCHS} lr={LR} seed={SEED}")

    train_tiles = tile_list("train")
    val_tiles = tile_list("val")
    log_line(logf, f"train tiles: {train_tiles}")
    log_line(logf, f"val tiles:   {val_tiles}")

    t0 = time.time()
    train_cache = TileCache(str(DATA), train_tiles)
    val_cache = TileCache(str(DATA), val_tiles)
    log_line(logf, f"Tile caches loaded in {time.time()-t0:.1f}s")

    train_ds = PatchDataset(train_cache, train_tiles, augment=True)
    val_ds = PatchDataset(val_cache, val_tiles, augment=False)
    log_line(logf, f"train patches: {len(train_ds)}  val patches: {len(val_ds)}")

    # exact pos_weight from full train-split pixel counts
    total_pix = sum(m.size for m in train_cache.masks.values())
    pos_pix = sum(int(m.sum()) for m in train_cache.masks.values())
    pos_weight = (total_pix - pos_pix) / pos_pix
    log_line(logf, f"train building-pixel fraction: {pos_pix/total_pix:.4f}  pos_weight: {pos_weight:.3f}")

    model = UNet(base=BASE_CHANNELS)
    log_line(logf, f"model params: {count_params(model):,}")
    loss_fn = CombinedLoss(pos_weight=pos_weight, boundary_weight=5.0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    val_loader_full = DataLoader(val_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    def val_iter_fn(n_batches):
        it = iter(val_loader_full)
        for i in range(n_batches):
            try:
                yield next(it)
            except StopIteration:
                return

    history = []
    best_iou = -1.0
    best_path = os.path.join(CKPT_DIR, "best_unet.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_t0 = time.time()
        train_iter = iter(train_loader)
        running = {"bce": 0.0, "dice": 0.0, "total": 0.0}
        n_seen = 0
        for step in range(STEPS_PER_EPOCH):
            try:
                xb, yb, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb, _ = next(train_iter)
            opt.zero_grad()
            logits = model(xb)
            loss, parts = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            for k in running:
                running[k] += parts[k]
            n_seen += 1
            if step % 30 == 0:
                log_line(logf, f"  epoch {epoch} step {step}/{STEPS_PER_EPOCH} "
                                f"loss={parts['total']:.4f} (bce={parts['bce']:.4f} dice={parts['dice']:.4f})")

        n_quick_val_batches = max(1, VAL_QUICK_N // BATCH)
        val_iou, val_dice = quick_eval(model, val_iter_fn, n_quick_val_batches)
        ep_time = time.time() - ep_t0

        row = {
            "epoch": epoch,
            "train_loss": running["total"] / n_seen,
            "train_bce": running["bce"] / n_seen,
            "train_dice_loss": running["dice"] / n_seen,
            "val_iou_quick": val_iou,
            "val_dice_quick": val_dice,
            "epoch_time_sec": ep_time,
        }
        history.append(row)
        log_line(logf, f"EPOCH {epoch}/{EPOCHS} done in {ep_time/60:.1f}min | "
                        f"train_loss={row['train_loss']:.4f} | "
                        f"quick val IoU={val_iou:.4f} Dice={val_dice:.4f}")

        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "last_unet.pt"))
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_path)
            log_line(logf, f"  -> new best (quick val IoU={best_iou:.4f}), saved {best_path}")

    with open(os.path.join(os.path.dirname(__file__), "..", "train_history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    meta = {
        "base_channels": BASE_CHANNELS, "batch": BATCH, "steps_per_epoch": STEPS_PER_EPOCH,
        "epochs": EPOCHS, "lr": LR, "pos_weight": pos_weight, "best_quick_val_iou": best_iou,
        "train_patches": len(train_ds), "val_patches": len(val_ds),
    }
    with open(os.path.join(os.path.dirname(__file__), "..", "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log_line(logf, f"\nTRAINING DONE. best quick-val IoU={best_iou:.4f}")
    logf.close()


if __name__ == "__main__":
    main()
