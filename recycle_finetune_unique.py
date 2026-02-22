import os, copy, argparse, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def count_per_class(ds):
    counts = [0] * len(ds.classes)
    for _, y in ds.samples:
        counts[y] += 1
    return counts


def make_class_weights(counts):
    c = torch.tensor(counts, dtype=torch.float32)
    w = 1.0 / torch.sqrt(c + 1.0)
    w = w / w.mean()
    return w


def mixup_batch(x, y, alpha=0.25):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = random.betavariate(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2, y2 = x[perm], y[perm]
    xm = lam * x + (1 - lam) * x2
    return xm, y, y2, lam


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, label_smoothing=0.03):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class RecycleNet(nn.Module):
    def __init__(self, num_classes, drop=0.35):
        super().__init__()
        self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_f = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_f, 384),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def set_trainable(model, stage):
    for p in model.net.features.parameters():
        p.requires_grad = False
    for p in model.net.classifier.parameters():
        p.requires_grad = True

    blocks = list(model.net.features.children())
    if stage == 1:
        for b in blocks[-2:]:
            for p in b.parameters():
                p.requires_grad = True
    elif stage == 2:
        for b in blocks[-4:]:
            for p in b.parameters():
                p.requires_grad = True
    elif stage == 3:
        for p in model.net.features.parameters():
            p.requires_grad = True


@torch.no_grad()
def eval_model(model, loader, loss_fn, num_classes):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits, y)
        preds = logits.argmax(1)

        for t, p in zip(y.cpu(), preds.cpu()):
            cm[t, p] += 1

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += (preds == y).float().mean().item() * bs
        n += bs

    f1s = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    macro_f1 = sum(f1s) / num_classes

    return total_loss / n, total_acc / n, macro_f1, cm


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def main(args):
    print("Starting training on:", DEVICE)

    train_tf, val_tf = build_transforms()
    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(args.data, "val"), transform=val_tf)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2)

    counts = count_per_class(train_ds)
    class_w = make_class_weights(counts).to(DEVICE)

    model = RecycleNet(num_classes=num_classes, drop=args.drop).to(DEVICE)
    loss_fn = FocalLoss(gamma=args.focal_gamma, weight=class_w, label_smoothing=0.03)

    best = {"f1": -1.0, "state": None, "epoch": -1, "cm": None}

    for ep in range(1, args.epochs + 1):
        if ep <= 3:
            stage, lr = 0, args.lr
        elif ep <= 7:
            stage, lr = 1, args.lr * 0.6
        elif ep <= 10:
            stage, lr = 2, args.lr * 0.35
        else:
            stage, lr = 3, args.lr * 0.20

        set_trainable(model, stage)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_loader))

        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            xm, y1, y2, lam = mixup_batch(x, y, alpha=args.mixup)

            opt.zero_grad()
            logits = model(xm)

            if y2 is None:
                loss = loss_fn(logits, y1)
            else:
                loss = lam * loss_fn(logits, y1) + (1 - lam) * loss_fn(logits, y2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += (logits.argmax(1) == y).float().mean().item() * bs
            n += bs

        tr_loss, tr_acc = total_loss / n, total_acc / n
        va_loss, va_acc, va_f1, cm = eval_model(model, val_loader, loss_fn, num_classes)

        print(
            f"Epoch {ep:02d} | stage {stage} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} macroF1 {va_f1:.3f}"
        )

        if va_f1 > best["f1"]:
            best.update({"f1": va_f1, "epoch": ep, "state": copy.deepcopy(model.state_dict()), "cm": cm.clone()})

    print("\nBest macroF1:", round(best["f1"], 4), "at epoch", best["epoch"])
    print("Classes:", train_ds.classes)
    print("Confusion Matrix:\n", best["cm"])

    torch.save({
        "model_state": best["state"],
        "classes": train_ds.classes,
        "best_epoch": best["epoch"],
        "best_macroF1": best["f1"],
        "class_counts": counts,
    }, args.out)

    print("Saved:", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--out", type=str, default="best_recycle_unique.pt")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--drop", type=float, default=0.35)
    p.add_argument("--mixup", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=1.5)
    args = p.parse_args()
    main(args)