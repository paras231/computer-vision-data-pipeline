"""
Fine-tune YOLOv8 on weapon_data2 (9-class weapon detection)
------------------------------------------------------------
Requires weapon_data2_dataloader_pipeline.py to be run first so that
weapon_data2_yolo_dataset/ and dataset.yaml are on disk.

Steps:
    1. Load pretrained YOLOv8s (small) — better capacity for 9 classes
    2. Fine-tune with cosine LR schedule + mosaic augmentation
    3. Evaluate on test split — per-class AP, mAP50, mAP50-95
    4. Visualise confusion matrix and per-class precision-recall curves
    5. Run inference on test images and save annotated outputs

Usage:
    python weapon_data2_dataloader_pipeline.py   # build dataset folder first
    python finetune_yolov8_weapon_data2.py

Requires:
    pip install ultralytics
"""

import torch
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

DATASET_YAML  = "weapon_data2_yolo_dataset/dataset.yaml"
BASE_MODEL    = "yolov8s.pt"           # small variant — better than nano for 9 classes
PROJECT_DIR   = "runs/weapon_data2"
RUN_NAME      = "finetune_v1"
IMG_SIZE      = 640
EPOCHS        = 10
BATCH_SIZE    = 16
LR0           = 5e-4                   # lower starting LR for multi-class fine-tune
LRF           = 0.01                   # final lr = LR0 * LRF (cosine decay)
WARMUP_EPOCHS = 3
PATIENCE      = 20                     # early-stopping patience (epochs)
FREEZE_LAYERS = 10                     # freeze first N backbone layers
CONF_THRESH   = 0.25                   # inference confidence threshold
IOU_THRESH    = 0.45                   # NMS IoU threshold

DEVICE = 0 if torch.cuda.is_available() else "cpu"

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

# ---------------------------------------------------------
# 1. Load pretrained YOLOv8s
#    Weights are downloaded automatically on first run (~22 MB)
# ---------------------------------------------------------

model = YOLO(BASE_MODEL)
print(f"Base model   : {BASE_MODEL}")
print(f"Device       : {DEVICE}")
print(f"Dataset YAML : {DATASET_YAML}")

# ---------------------------------------------------------
# 2. Fine-tune
#
#    Key choices vs. gun fine-tune:
#      - yolov8s instead of yolov8n (more capacity for 9 classes)
#      - lower LR0 (5e-4) to avoid overwriting learned features
#      - cos_lr=True for smooth decay on longer training run
#      - mosaic=1.0 + mixup=0.1 for better generalisation across classes
#      - close_mosaic=10 disables mosaic in final 10 epochs (stabilises predictions)
# ---------------------------------------------------------

results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    lr0=LR0,
    lrf=LRF,
    warmup_epochs=WARMUP_EPOCHS,
    cos_lr=True,
    momentum=0.937,
    weight_decay=5e-4,
    patience=PATIENCE,
    freeze=FREEZE_LAYERS,
    # augmentation
    mosaic=1.0,
    mixup=0.1,
    close_mosaic=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5,
    # output
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
    plots=True,
    save=True,
    device=DEVICE,
    workers=0,
    verbose=True,
)

print(f"\nTraining complete.")
print(f"Best weights : {results.save_dir}/weights/best.pt")
print(f"Last weights : {results.save_dir}/weights/last.pt")

# ---------------------------------------------------------
# 3. Evaluate on test split using best checkpoint
# ---------------------------------------------------------

best_weights = Path(results.save_dir) / "weights" / "best.pt"
model_best   = YOLO(str(best_weights))

print("\n--- Test Set Evaluation ---")
metrics = model_best.val(
    data=DATASET_YAML,
    split="test",
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    conf=CONF_THRESH,
    iou=IOU_THRESH,
    device=DEVICE,
    workers=0,
    verbose=True,
    plots=True,                    # saves confusion matrix + PR curves
    project=PROJECT_DIR,
    name=f"{RUN_NAME}_test_eval",
    exist_ok=True,
)

print(f"\n{'Metric':<20} {'Value':>8}")
print("-" * 30)
print(f"{'mAP50':<20} {metrics.box.map50:>8.4f}")
print(f"{'mAP50-95':<20} {metrics.box.map:>8.4f}")
print(f"{'Precision':<20} {metrics.box.mp:>8.4f}")
print(f"{'Recall':<20} {metrics.box.mr:>8.4f}")

# Per-class AP50
print(f"\n--- Per-Class AP50 ---")
if metrics.box.ap_class_index is not None:
    for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
        cls_name = CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else str(idx)
        print(f"  {cls_name:<22} {ap:.4f}")

# ---------------------------------------------------------
# 4. Inference on test images — save annotated outputs
# ---------------------------------------------------------

test_images_dir = Path("weapon_data2_yolo_dataset/images/test")
test_images     = list(test_images_dir.iterdir())

if test_images:
    print(f"\nRunning inference on {len(test_images)} test image(s)...")
    preds = model_best.predict(
        source=str(test_images_dir),
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=True,
        save_txt=True,             # also saves predicted labels as .txt
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_inference",
        exist_ok=True,
        workers=0,
    )

    print(f"\n{'Image':<40} {'Detections':>10}")
    print("-" * 52)
    for r in preds:
        n_det = len(r.boxes) if r.boxes is not None else 0
        classes_found = []
        if r.boxes is not None and len(r.boxes):
            for cls_id in r.boxes.cls.int().tolist():
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                if cls_name not in classes_found:
                    classes_found.append(cls_name)
        cls_str = ", ".join(classes_found) if classes_found else "none"
        print(f"  {Path(r.path).name:<38} {n_det:>4}  [{cls_str}]")

    print(f"\nAnnotated images saved to: {PROJECT_DIR}/{RUN_NAME}_inference/")

print("\nDone.")
