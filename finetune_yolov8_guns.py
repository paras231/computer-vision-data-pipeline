"""
Fine-tune YOLOv8 on the Gun Detection Dataset
----------------------------------------------
Builds directly on gun_dataloader_pipeline.py (must be run first so that
gun_yolo_dataset/ and dataset.yaml are present on disk).

Steps:
1. Load pretrained YOLOv8n (nano) — smallest variant, good starting point
2. Fine-tune on gun_yolo_dataset using the generated dataset.yaml
3. Evaluate on the test split — mAP50, mAP50-95, precision, recall
4. Run inference on a few sample images and save annotated outputs

Usage:
    python gun_dataloader_pipeline.py   # builds the dataset folder first
    python finetune_yolov8_guns.py

Requires:
    pip install ultralytics
"""

import os
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

DATASET_YAML   = "gun_yolo_dataset/dataset.yaml"
BASE_MODEL     = "yolov8n.pt"          # pretrained nano checkpoint (auto-downloaded)
PROJECT_DIR    = "runs/gun_detection"
RUN_NAME       = "finetune_v1"
IMG_SIZE       = 640
EPOCHS         = 50
BATCH_SIZE     = 4                     # small dataset — keep batch small
LR0            = 1e-3                  # initial learning rate
LRF            = 0.01                  # final lr = LR0 * LRF
PATIENCE       = 15                    # early stopping patience
FREEZE_LAYERS  = 10                    # freeze first N backbone layers

# ---------------------------------------------------------
# 1. Load pretrained YOLOv8n
#    Weights are downloaded automatically on first run
# ---------------------------------------------------------

model = YOLO(BASE_MODEL)
print(f"Loaded base model: {BASE_MODEL}")
print(f"Task: {model.task} | Classes: {model.names}")

# ---------------------------------------------------------
# 2. Fine-tune
#    freeze=N keeps the first N backbone layers frozen (feature extractor),
#    mirroring the frozen-early-layers strategy in finetune_resnet.py
# ---------------------------------------------------------

results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    lr0=LR0,
    lrf=LRF,
    patience=PATIENCE,
    freeze=FREEZE_LAYERS,
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
    plots=True,         # saves loss/metric curves
    save=True,          # saves best.pt + last.pt
    device=0 if __import__("torch").cuda.is_available() else "cpu",
    workers=0,          # 0 avoids multiprocessing issues on Windows
    verbose=True,
)

print(f"\nTraining complete.")
print(f"Best weights saved to: {results.save_dir}/weights/best.pt")

# ---------------------------------------------------------
# 3. Evaluate on test split using best saved checkpoint
# ---------------------------------------------------------

best_weights = Path(results.save_dir) / "weights" / "best.pt"
model_best = YOLO(str(best_weights))

print("\nEvaluating on test split...")
metrics = model_best.val(
    data=DATASET_YAML,
    split="test",
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=0 if __import__("torch").cuda.is_available() else "cpu",
    workers=0,
    verbose=True,
)

print(f"\n--- Test Results ---")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")

# ---------------------------------------------------------
# 4. Inference on test images — save annotated outputs
#    (mirrors the confusion matrix / visualisation step in finetune_resnet.py)
# ---------------------------------------------------------

test_images_dir = Path("gun_yolo_dataset/images/test")
test_images = list(test_images_dir.glob("*"))

if test_images:
    print(f"\nRunning inference on {len(test_images)} test image(s)...")
    inference_results = model_best.predict(
        source=str(test_images_dir),
        imgsz=IMG_SIZE,
        conf=0.25,
        save=True,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_inference",
        exist_ok=True,
        workers=0,
    )

    for r in inference_results:
        n_det = len(r.boxes) if r.boxes is not None else 0
        print(f"  {Path(r.path).name}: {n_det} detection(s)")
    print(f"\nAnnotated images saved to: {PROJECT_DIR}/{RUN_NAME}_inference/")

print("\nDone.")
