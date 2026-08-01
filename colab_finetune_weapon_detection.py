"""
Weapon Detection — YOLOv8 Fine-Tune (Google Colab / GPU)
=========================================================
Copy-paste this entire file into a Colab notebook cell, or upload and run:
    !python colab_finetune_weapon_detection.py

Before running:
    1. Upload your zipped dataset to Colab (e.g. weapon_dataset.zip)
    2. Set DATASET_ZIP below to the exact filename you uploaded
    3. Runtime → Change runtime type → GPU (T4 recommended)

Dataset zip expected structure
-------------------------------
weapon_dataset.zip
└── weapon_data2_yolo_dataset/
    ├── dataset.yaml
    ├── images/
    │   ├── train/  *.jpg / *.png
    │   ├── val/    *.jpg / *.png
    │   └── test/   *.jpg / *.png
    └── labels/
        ├── train/  *.txt
        ├── val/    *.txt
        └── test/   *.txt

dataset.yaml must contain absolute paths OR relative paths pointing to
the unzipped folder. The script patches the yaml automatically.
"""

# ============================================================
# STEP 0 — Install dependencies (Colab only needs ultralytics)
# ============================================================
import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip_install("ultralytics")

# ============================================================
# STEP 1 — Unzip dataset
# ============================================================
import zipfile, os, shutil
from pathlib import Path

# ── USER CONFIG ──────────────────────────────────────────────
DATASET_ZIP   = "weapon_dataset.zip"          # filename you uploaded to Colab
EXTRACT_DIR   = "/content/weapon_dataset"     # where to unzip
DATASET_YAML  = None                          # set to override auto-detect
BASE_MODEL    = "yolov8m.pt"                  # n / s / m / l / x
PROJECT_DIR   = "/content/runs/weapon"
RUN_NAME      = "finetune_v2_colab"
IMG_SIZE      = 640
EPOCHS        = 50
BATCH_SIZE    = 16                            # lower to 8 if OOM
LR0           = 3e-4
LRF           = 0.01
WARMUP_EPOCHS = 3
PATIENCE      = 15
FREEZE_LAYERS = 10                            # freeze first N backbone layers
CONF_THRESH   = 0.25
IOU_THRESH    = 0.45
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  Weapon Detection — YOLOv8 Fine-Tune (Colab GPU)")
print("=" * 60)

zip_path = Path(DATASET_ZIP)
if not zip_path.exists():
    raise FileNotFoundError(
        f"\n[ERROR] '{DATASET_ZIP}' not found.\n"
        "Upload your dataset zip to Colab first, then re-run.\n"
        "Files panel → Upload icon  (or drag-and-drop to /content/)"
    )

extract_path = Path(EXTRACT_DIR)
if extract_path.exists():
    shutil.rmtree(extract_path)   # clean previous run
extract_path.mkdir(parents=True)

print(f"\n[1/5] Unzipping {DATASET_ZIP} → {EXTRACT_DIR} ...")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_path)
print("      Done.")

# ============================================================
# STEP 2 — Auto-detect dataset.yaml and patch paths
# ============================================================
import yaml

if DATASET_YAML is None:
    yamls = list(extract_path.rglob("dataset.yaml"))
    if not yamls:
        raise FileNotFoundError(
            "[ERROR] No dataset.yaml found inside the zip.\n"
            "Expected path: weapon_data2_yolo_dataset/dataset.yaml"
        )
    yaml_path = yamls[0]
else:
    yaml_path = Path(DATASET_YAML)

print(f"\n[2/5] Found dataset.yaml at: {yaml_path}")

# Patch train / val / test paths to absolute Colab paths
dataset_root = yaml_path.parent

with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

# Overwrite path with absolute root so YOLO resolves splits correctly
cfg["path"] = str(dataset_root)
for split in ("train", "val", "test"):
    if split in cfg:
        rel = cfg[split]
        # If already absolute and exists, leave it; otherwise make relative to root
        if not Path(rel).is_absolute():
            cfg[split] = rel   # YOLO joins cfg["path"] + cfg[split] automatically

with open(yaml_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"      Patched dataset root → {dataset_root}")
print(f"      Classes ({cfg.get('nc', '?')}): {cfg.get('names', [])}")

# ============================================================
# STEP 3 — GPU check
# ============================================================
import torch

device = 0 if torch.cuda.is_available() else "cpu"

print(f"\n[3/5] Device: ", end="")
if torch.cuda.is_available():
    print(f"GPU — {torch.cuda.get_device_name(0)}")
    print(f"      VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("CPU  ⚠  Training on CPU will be very slow. Enable GPU in Colab runtime settings.")

CLASS_NAMES = cfg.get("names", [])
NC          = cfg.get("nc", len(CLASS_NAMES))

# ============================================================
# STEP 4 — Fine-tune
# ============================================================
from ultralytics import YOLO

print(f"\n[4/5] Loading base model: {BASE_MODEL}  (downloads automatically if needed)")
model = YOLO(BASE_MODEL)

print(f"\n      Starting training — {EPOCHS} epochs, batch {BATCH_SIZE}, img {IMG_SIZE}px")
print(f"      Results → {PROJECT_DIR}/{RUN_NAME}/\n")

results = model.train(
    data=str(yaml_path),
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
    mixup=0.15,
    close_mosaic=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,
    # output
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
    plots=True,
    save=True,
    device=device,
    workers=2,
    verbose=True,
    amp=True,               # mixed precision — faster on T4/A100
)

best_weights = Path(results.save_dir) / "weights" / "best.pt"
last_weights = Path(results.save_dir) / "weights" / "last.pt"

print(f"\n      Training complete.")
print(f"      Best weights : {best_weights}")
print(f"      Last weights : {last_weights}")

# ============================================================
# STEP 5 — Evaluate on test split
# ============================================================
print(f"\n[5/5] Evaluating on test split ...")

model_best = YOLO(str(best_weights))

metrics = model_best.val(
    data=str(yaml_path),
    split="test",
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    conf=CONF_THRESH,
    iou=IOU_THRESH,
    device=device,
    workers=2,
    verbose=True,
    plots=True,
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

if metrics.box.ap_class_index is not None:
    print(f"\n--- Per-Class AP50 ---")
    for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
        name = CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else str(idx)
        print(f"  {name:<24} {ap:.4f}")

# ============================================================
# STEP 6 — Inference on test images (sample of 20)
# ============================================================
test_img_dir = dataset_root / "images" / "test"
if test_img_dir.exists():
    test_imgs = sorted(test_img_dir.iterdir())[:20]
    if test_imgs:
        print(f"\n--- Inference on {len(test_imgs)} sample test image(s) ---")
        preds = model_best.predict(
            source=str(test_img_dir),
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            save=True,
            save_txt=True,
            project=PROJECT_DIR,
            name=f"{RUN_NAME}_inference",
            exist_ok=True,
            workers=2,
        )
        print(f"\n{'Image':<40} {'Dets':>5}  Classes")
        print("-" * 60)
        for r in preds:
            n = len(r.boxes) if r.boxes is not None else 0
            found = []
            if r.boxes is not None:
                for cid in r.boxes.cls.int().tolist():
                    cn = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
                    if cn not in found:
                        found.append(cn)
            print(f"  {Path(r.path).name:<38} {n:>5}  {', '.join(found) or 'none'}")
        print(f"\nAnnotated images → {PROJECT_DIR}/{RUN_NAME}_inference/")

# ============================================================
# STEP 7 — Download best weights from Colab
# ============================================================
print("\n" + "=" * 60)
print("  Training finished!")
print(f"  Best model : {best_weights}")
print("=" * 60)

try:
    from google.colab import files as colab_files
    print("\nDownloading best.pt to your local machine ...")
    colab_files.download(str(best_weights))
except ImportError:
    print("\n[INFO] Not running in Colab — copy weights manually:")
    print(f"       {best_weights}")
