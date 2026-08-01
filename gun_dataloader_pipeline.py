"""
Gun Detection (YOLOv8 format) - Dataset + Dataloader Pipeline
--------------------------------------------------------------
Mirrors dataloader_pipeline.py but follows YOLO object-detection architecture.

Steps:
1. Parse gun_annotated into a dataframe (image path + label path per row)
2. Random train / val / test split
3. Build the YOLOv8 directory structure and write dataset.yaml
4. Return Ultralytics dataloaders for use in fine-tuning
5. Sanity check — inspect a batch

Requires:
    pip install ultralytics albumentations
"""

import os
import shutil
import random
import yaml
import pandas as pd
from pathlib import Path

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

SRC_IMAGES_DIR = "gun_annotated/images"
SRC_LABELS_DIR = "gun_annotated/labels"
DATASET_ROOT   = "gun_yolo_dataset"     # output directory YOLOv8 will read from
CLASS_NAMES    = ["gun"]
IMG_SIZE       = 640                    # standard YOLO input resolution
BATCH_SIZE     = 4                      # small — only 15 images total
SEED           = 42

random.seed(SEED)

# ---------------------------------------------------------
# 1. Build a dataframe — one row per image that has a matching label
# ---------------------------------------------------------

def build_dataframe(images_dir: str, labels_dir: str) -> pd.DataFrame:
    records = []
    for label_file in sorted(Path(labels_dir).glob("*.txt")):
        stem = label_file.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = Path(images_dir) / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"Warning: no image found for {label_file.name}")
            continue
        records.append({"image": str(img_path), "label": str(label_file)})
    return pd.DataFrame(records)


df = build_dataframe(SRC_IMAGES_DIR, SRC_LABELS_DIR)
print(f"Total paired samples: {len(df)}")
print(df[["image", "label"]].to_string(index=False))

# ---------------------------------------------------------
# 2. Train / val / test split
#    (no stratification needed — single-class detection)
# ---------------------------------------------------------

indices = df.index.tolist()
random.shuffle(indices)

n_total = len(indices)
n_val   = max(1, round(n_total * 0.15))
n_test  = max(1, round(n_total * 0.10))
n_train = n_total - n_val - n_test

train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]

train_df = df.loc[train_idx].reset_index(drop=True)
val_df   = df.loc[val_idx].reset_index(drop=True)
test_df  = df.loc[test_idx].reset_index(drop=True)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---------------------------------------------------------
# 3. Build YOLOv8 directory structure
#
#    gun_yolo_dataset/
#      images/train/  images/val/  images/test/
#      labels/train/  labels/val/  labels/test/
#      dataset.yaml
# ---------------------------------------------------------

def populate_split(split_df: pd.DataFrame, split: str, root: str) -> None:
    img_out = Path(root) / "images" / split
    lbl_out = Path(root) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for _, row in split_df.iterrows():
        shutil.copy(row["image"], img_out / Path(row["image"]).name)
        shutil.copy(row["label"], lbl_out / Path(row["label"]).name)


populate_split(train_df, "train", DATASET_ROOT)
populate_split(val_df,   "val",   DATASET_ROOT)
populate_split(test_df,  "test",  DATASET_ROOT)

# ---------------------------------------------------------
# 4. Write dataset.yaml — the single config file YOLOv8 needs
# ---------------------------------------------------------

dataset_yaml_path = Path(DATASET_ROOT) / "dataset.yaml"
dataset_cfg = {
    "path":  str(Path(DATASET_ROOT).resolve()),
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc":    len(CLASS_NAMES),
    "names": CLASS_NAMES,
}
with open(dataset_yaml_path, "w") as f:
    yaml.dump(dataset_cfg, f, default_flow_style=False)

print(f"\nDataset YAML written to: {dataset_yaml_path}")
print(yaml.dump(dataset_cfg, default_flow_style=False))

# ---------------------------------------------------------
# 5. Build Ultralytics YOLODataset + DataLoader for each split
#    (mirrors the ImageDataset + DataLoader section from dataloader_pipeline.py)
# ---------------------------------------------------------

def build_yolo_loader(split: str, augment: bool) -> DataLoader:
    img_dir = str(Path(DATASET_ROOT) / "images" / split)
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
        num_workers=0,                  # 0 keeps it portable on Windows
        collate_fn=YOLODataset.collate_fn,
    )


train_loader = build_yolo_loader("train", augment=True)
val_loader   = build_yolo_loader("val",   augment=False)
test_loader  = build_yolo_loader("test",  augment=False)

# ---------------------------------------------------------
# 6. Sanity check — pull one batch and inspect shapes
# ---------------------------------------------------------

if __name__ == "__main__":
    batch = next(iter(train_loader))
    imgs   = batch["img"]
    bboxes = batch["bboxes"]
    clss   = batch["cls"]

    print(f"\nBatch image tensor shape : {imgs.shape}")     # [B, 3, H, W]
    print(f"Batch bounding boxes      : {bboxes.shape}")   # [N_objects, 4] — xyxy normalized
    print(f"Batch class ids           : {clss.shape}")     # [N_objects, 1]
    print(f"\nPipeline ready. train_loader / val_loader / test_loader "
          f"can be passed into finetune_yolov8_guns.py")
