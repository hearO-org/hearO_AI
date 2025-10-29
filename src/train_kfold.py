import os, yaml, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.collate import pad_collate
from src.utils.seed import set_seed
from src.utils.metrics import accuracy
from src.utils.augment import SpecAugment
from src.datasets.us8k import UrbanSoundSubset
from src.models.cnn_small import CNN_Small

def get_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, loader, optim, device, criterion):
    model.train()
    accs, losses = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optim.zero_grad(); loss.backward(); optim.step()
        losses.append(loss.item()); accs.append(accuracy(logits, y))
    return np.mean(losses), np.mean(accs)

@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    accs, losses = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item()); accs.append(accuracy(logits, y))
    return np.mean(losses), np.mean(accs)

def main(cfg_path="./configs/config.yaml"):
    cfg = get_cfg(cfg_path)
    set_seed(cfg.get("seed", 42))

    # --- (1) 안전한 타입 캐스팅 ---
    cfg["train"]["lr"] = float(cfg["train"]["lr"])
    cfg["train"]["weight_decay"] = float(cfg["train"]["weight_decay"])
    cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"])
    cfg["train"]["epochs"] = int(cfg["train"]["epochs"])
    cfg["train"]["num_workers"] = int(cfg["train"]["num_workers"])
    cfg["train"]["early_stop_patience"] = int(cfg["train"]["early_stop_patience"])
    cfg["sample_rate"] = int(cfg["sample_rate"])
    cfg["n_mels"] = int(cfg["n_mels"])
    cfg["win_ms"] = int(cfg["win_ms"])
    cfg["hop_ms"] = int(cfg["hop_ms"])

    # --- (2) device 단일화 + 폴백 ---
    req = cfg.get("device", "cpu")
    if req == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable. Falling back to CPU.")
        req = "cpu"
    device = torch.device(req)
    print(f"[INFO] Using device: {device}")

    data_root = cfg["data_root"]
    meta_csv  = os.path.join(data_root, cfg["metadata_csv"])
    audio_root= os.path.join(data_root, "audio")
    classes   = cfg["class_list"]
    num_cls   = len(classes)

    os.makedirs(cfg["out_dir"], exist_ok=True)

    fold_ids = list(range(1, 11))
    fold_scores = []

    for test_fold in fold_ids:
        train_folds = [f for f in fold_ids if f != test_fold]

        aug = SpecAugment(cfg["train"]["specaug_time_mask"], cfg["train"]["specaug_freq_mask"])

        ds_train = UrbanSoundSubset(
            meta_csv=meta_csv, audio_root=audio_root, class_list=classes,
            folds=train_folds, sr=cfg["sample_rate"], n_mels=cfg["n_mels"],
            win_ms=cfg["win_ms"], hop_ms=cfg["hop_ms"], augment=aug)

        ds_val = UrbanSoundSubset(
            meta_csv=meta_csv, audio_root=audio_root, class_list=classes,
            folds=[test_fold], sr=cfg["sample_rate"], n_mels=cfg["n_mels"],
            win_ms=cfg["win_ms"], hop_ms=cfg["hop_ms"], augment=None)

        dl_train = DataLoader(
            ds_train,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["train"]["num_workers"],
            pin_memory=(device.type == "cuda"),
            collate_fn=pad_collate,  # ★ 추가
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["train"]["num_workers"],
            pin_memory=(device.type == "cuda"),
            collate_fn=pad_collate,  # ★ 추가
        )

        mcfg = cfg["model"]
        model = CNN_Small(
            in_ch=mcfg["in_channels"], num_classes=num_cls,
            num_filters=tuple(mcfg["num_filters"]), dropout=mcfg["dropout"]
        ).to(device)

        optim = torch.optim.Adam(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()

        best_val, best_state = -1.0, None
        patience, p = cfg["train"]["early_stop_patience"], 0

        for epoch in range(1, cfg["train"]["epochs"] + 1):
            tr_loss, tr_acc = train_one_epoch(model, dl_train, optim, device, criterion)
            va_loss, va_acc = eval_epoch(model, dl_val, device, criterion)

            print(f"[Fold {test_fold}] Epoch {epoch:02d} | "
                  f"train acc {tr_acc:.3f} loss {tr_loss:.3f} | "
                  f"val acc {va_acc:.3f} loss {va_loss:.3f}")

            if va_acc > best_val:
                best_val, p = va_acc, 0
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                p += 1
                if p >= patience:
                    print(f"[Fold {test_fold}] Early stop at epoch {epoch}")
                    break

        fold_ckpt = os.path.join(cfg["out_dir"], f"best_fold{test_fold}.pt")
        torch.save(best_state, fold_ckpt)
        fold_scores.append(best_val)
        print(f"[Fold {test_fold}] BEST VAL ACC = {best_val:.4f} | saved {fold_ckpt}")

    scores = np.array(fold_scores)
    report = f"10-Fold ACC mean={scores.mean():.4f}, std={scores.std():.4f}\nPer-fold: {np.round(scores,4)}"
    print(report)
    with open(os.path.join(cfg["out_dir"], "kfold_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    plt.figure()
    plt.boxplot(scores)
    plt.title("UrbanSound Commercial Subset - 10-Fold Accuracy")
    plt.savefig(os.path.join(cfg["out_dir"], "kfold_boxplot.png"), dpi=150)
    print(f"[OK] Saved boxplot to {os.path.join(cfg['out_dir'], 'kfold_boxplot.png')}")

if __name__ == "__main__":
    main()
