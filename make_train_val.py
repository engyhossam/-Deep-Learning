import random, shutil, os
from pathlib import Path

random.seed(42)
SRC_TRAIN = Path(r"archive/1024+/entrainement")
OUT = Path("data")
VAL_RATIO = 0.2

def make_dir(p): p.mkdir(parents=True, exist_ok=True)

def fast_copy(src: Path, dst: Path):
    try:
        if dst.exists(): return
        os.link(src, dst)  # faster than copy if same drive
    except Exception:
        if not dst.exists():
            shutil.copy2(src, dst)

def main():
    if not SRC_TRAIN.exists():
        raise FileNotFoundError(f"Not found: {SRC_TRAIN}")

    for split in ["train", "val"]:
        make_dir(OUT / split)

    classes = [d for d in SRC_TRAIN.iterdir() if d.is_dir()]
    if not classes:
        raise ValueError("No class folders found inside entrainement")

    for c in classes:
        imgs = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            imgs += list(c.glob(ext))

        if len(imgs) == 0:
            print(f"Warning: no images in {c.name}")
            continue

        random.shuffle(imgs)
        cut = int(len(imgs) * (1 - VAL_RATIO))
        train_imgs, val_imgs = imgs[:cut], imgs[cut:]

        train_dir = OUT / "train" / c.name
        val_dir   = OUT / "val" / c.name
        make_dir(train_dir); make_dir(val_dir)

        for i, p in enumerate(train_imgs, 1):
            fast_copy(p, train_dir / p.name)
            if i % 400 == 0:
                print(f"{c.name} train: {i}/{len(train_imgs)}")

        for i, p in enumerate(val_imgs, 1):
            fast_copy(p, val_dir / p.name)
            if i % 400 == 0:
                print(f"{c.name} val: {i}/{len(val_imgs)}")

        print(f"✅ Finished class: {c.name}")

    print("✅ Done. data/train and data/val are ready.")

if __name__ == "__main__":
    main()