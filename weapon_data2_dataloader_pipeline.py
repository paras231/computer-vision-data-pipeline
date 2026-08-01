"""
Weapon Detection (weapon_data2) — YOLOv8 Dataset + DataLoader Pipeline
------------------------------------------------------------------------
Dataset layout:
    weapon_data2/
        metadata.csv              (imagefile, labelfile, target, train_id)
        weapon_detection/
            train/images/  train/labels/   (571 images, train_id == 1)
            val/images/    val/labels/     (143 images, train_id == 0)

Pipeline steps:
    1. Parse metadata.csv into a DataFrame with resolved file paths
    2. Use train_id to split: train (train_id=1), then divide the val
       pool (train_id=0) into val (80 %) and test (20 %)
    3. Build the YOLOv8 directory structure and write dataset.yaml
    4. Construct Ultralytics YOLODataset + DataLoader for each split
    5. Sanity-check a batch

9 weapon classes:
    0 Automatic Rifle | 1 Bazooka | 2 Grenade Launcher | 3 Handgun
    4 Knife           | 5 Shotgun | 6 SMG               | 7 Sniper
    8 Sword

Requires:
    pip install ultralytics albumentations
"""

import os
import random
import shutil
import yaml
import pandas as pd
from pathlib import Path

from ultralytics.data.dataset import YOLODataset
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

METADATA_CSV   = Path("weapon_data2/metadata.csv")
SRC_TRAIN_DIR  = Path("weapon_data2/weapon_detection/train")
SRC_VAL_DIR    = Path("weapon_data2/weapon_detection/val")
DATASET_ROOT   = Path("weapon_data2_yolo_dataset")

CLASS_NAMES = [
    "Automatic Rifle",
    "Bazooka",
    "Grenade Launcher",
    "Handgun",
    "Knife",
    "Shotgun",
    "SMG",
    "Sniper",
    "Sword",
]

IMG_SIZE   = 640
BATCH_SIZE = 16
SEED       = 42

random.seed(SEED)

# ---------------------------------------------------------
# 1. Build DataFrame with resolved paths
# ---------------------------------------------------------

def build_dataframe() -> pd.DataFrame:
    df = pd.read_csv(METADATA_CSV)

    records = []
    for _, row in df.iterrows():
        src_dir = SRC_TRAIN_DIR if row["train_id"] == 1 else SRC_VAL_DIR
        img_path = src_dir / "images" / row["imagefile"]
        lbl_path = src_dir / "labels" / row["labelfile"]

        if not img_path.exists():
            print(f"Warning: missing image {img_path}")
            continue
        if not lbl_path.exists():
            print(f"Warning: missing label {lbl_path}")
            continue

        records.append({
            "image":    str(img_path),
            "label":    str(lbl_path),
            "class_id": int(row["target"]),
            "train_id": int(row["train_id"]),
        })

    return pd.DataFrame(records)


df = build_dataframe()
print(f"Total valid samples : {len(df)}")
print(f"  train_id=1 (train): {(df['train_id'] == 1).sum()}")
print(f"  train_id=0 (val pool): {(df['train_id'] == 0).sum()}")
print(f"\nClass distribution:\n{df['class_id'].value_counts().sort_index().rename(index={i: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))})}")

# ---------------------------------------------------------
# 2. Train / val / test split
#    train_id=1  → train
#    train_id=0  → shuffle, then 80 % val / 20 % test
# ---------------------------------------------------------

train_df   = df[df["train_id"] == 1].reset_index(drop=True)
val_pool   = df[df["train_id"] == 0].sample(frac=1, random_state=SEED).reset_index(drop=True)

n_val      = max(1, round(len(val_pool) * 0.80))
val_df     = val_pool.iloc[:n_val].reset_index(drop=True)
test_df    = val_pool.iloc[n_val:].reset_index(drop=True)

print(f"\nSplit — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---------------------------------------------------------
# 3. Build YOLOv8 directory structure
#
#    weapon_data2_yolo_dataset/
#      images/train/  images/val/  images/test/
#      labels/train/  labels/val/  labels/test/
#      dataset.yaml
# ---------------------------------------------------------

def populate_split(split_df: pd.DataFrame, split: str) -> None:
    img_out = DATASET_ROOT / "images" / split
    lbl_out = DATASET_ROOT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for _, row in split_df.iterrows():
        shutil.copy(row["image"], img_out / Path(row["image"]).name)
        shutil.copy(row["label"], lbl_out / Path(row["label"]).name)


if DATASET_ROOT.exists():
    shutil.rmtree(DATASET_ROOT)

populate_split(train_df, "train")
populate_split(val_df,   "val")
populate_split(test_df,  "test")

print(f"\nDataset written to: {DATASET_ROOT.resolve()}")

# ---------------------------------------------------------
# 4. Write dataset.yaml
# ---------------------------------------------------------

dataset_cfg = {
    "path":  str(DATASET_ROOT.resolve()),
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc":    len(CLASS_NAMES),
    "names": CLASS_NAMES,
}

yaml_path = DATASET_ROOT / "dataset.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(dataset_cfg, f, default_flow_style=False, allow_unicode=True)

print(f"dataset.yaml written to: {yaml_path}")
print(yaml.dump(dataset_cfg, default_flow_style=False, allow_unicode=True))

# ---------------------------------------------------------
# 5. Build Ultralytics YOLODataset + DataLoader per split
# ---------------------------------------------------------

def build_yolo_loader(split: str, augment: bool) -> DataLoader:
    img_dir = str(DATASET_ROOT / "images" / split)
    dataset = YOLODataset(
        img_path=img_dir,
        data={"nc": len(CLASS_NAMES), "names": CLASS_NAMES},
        imgsz=IMG_SIZE,
        augment=augment,
        task="detect",
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=0,          # 0 for Windows compatibility
        collate_fn=YOLODataset.collate_fn,
    )


train_loader = build_yolo_loader("train", augment=True)
val_loader   = build_yolo_loader("val",   augment=False)
test_loader  = build_yolo_loader("test",  augment=False)

# ---------------------------------------------------------
# 6. Sanity check
# ---------------------------------------------------------

if __name__ == "__main__":
    batch  = next(iter(train_loader))
    imgs   = batch["img"]
    bboxes = batch["bboxes"]
    clss   = batch["cls"]

    print(f"\nBatch image tensor : {imgs.shape}")    # [B, 3, H, W]
    print(f"Bounding boxes     : {bboxes.shape}")   # [N_objects, 4] xywh normalised
    print(f"Class ids          : {clss.shape}")     # [N_objects, 1]

    unique_cls = clss.unique().long().tolist()
    print(f"Classes in batch   : {[CLASS_NAMES[c] for c in unique_cls]}")

    print(f"\nLoaders ready:")
    print(f"  train_loader — {len(train_loader)} batches")
    print(f"  val_loader   — {len(val_loader)} batches")
    print(f"  test_loader  — {len(test_loader)} batches")
    print("\nPass dataset.yaml to YOLOv8 training:")
    print(f"  model.train(data='{yaml_path}', epochs=50, imgsz={IMG_SIZE})")
