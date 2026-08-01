"""
COCO128 (YOLO format) - EDA and Bounding Box Visualization
-------------------------------------------------------------
Reads YOLO-format label .txt files, draws boxes on images, and reports
class distribution and basic dataset stats.

YOLO label format (one line per object in each .txt file):
    class_id x_center y_center width height
    (all values normalized between 0 and 1, relative to image size)
"""

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import random

IMAGES_DIR = "coco128/coco128/images/train2017"
LABELS_DIR = "coco128/coco128/labels/train2017"

# COCO class names (index must match class_id in label files)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# ---------------------------------------------------------
# 1. Parse all label files into a dataframe (one row per object)
# ---------------------------------------------------------

def parse_labels(images_dir, labels_dir):
    records = []
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    for lf in label_files:
        img_name = os.path.splitext(lf)[0]
        # try common extensions
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(images_dir, img_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"Warning: no matching image found for label {lf}")
            continue

        with open(os.path.join(labels_dir, lf), "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            # image with no objects labeled
            records.append({
                "image": img_path, "label_file": lf,
                "class_id": None, "x_center": None, "y_center": None,
                "width": None, "height": None
            })
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_c, y_c, w, h = parts
            records.append({
                "image": img_path,
                "label_file": lf,
                "class_id": int(class_id),
                "x_center": float(x_c),
                "y_center": float(y_c),
                "width": float(w),
                "height": float(h),
            })

    return pd.DataFrame(records)

df = parse_labels(IMAGES_DIR, LABELS_DIR)
print(f"Total labeled objects: {df['class_id'].notna().sum()}")
print(f"Total images: {df['image'].nunique()}")

# ---------------------------------------------------------
# 2. Class distribution (per OBJECT, not per image)
# ---------------------------------------------------------

valid = df.dropna(subset=["class_id"])
valid = valid.copy()
valid["class_name"] = valid["class_id"].apply(lambda i: COCO_CLASSES[i] if i < len(COCO_CLASSES) else f"unknown_{i}")

class_counts = valid["class_name"].value_counts()
print("\nObject counts per class:")
print(class_counts)

class_counts.plot(kind="bar", figsize=(12, 5), title="Object count per class (COCO128)")
plt.tight_layout()
plt.savefig("coco128_class_distribution.png")
plt.close()
print("Saved coco128_class_distribution.png")

# Images with zero objects labeled (worth knowing about)
empty_images = df[df["class_id"].isna()]["image"].nunique()
print(f"\nImages with no labeled objects: {empty_images}")

# Objects per image distribution (some images have many, some have few)
objects_per_image = valid.groupby("image").size()
print(f"\nObjects per image - min: {objects_per_image.min()}, "
      f"max: {objects_per_image.max()}, mean: {objects_per_image.mean():.2f}")

# ---------------------------------------------------------
# 3. Draw bounding boxes on a few sample images to visually verify
# ---------------------------------------------------------

def draw_boxes(image_path, boxes_df):
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    for _, row in boxes_df.iterrows():
        if pd.isna(row["class_id"]):
            continue
        # Convert normalized YOLO format -> pixel coordinates
        x_c, y_c, w, h = row["x_center"], row["y_center"], row["width"], row["height"]
        x1 = int((x_c - w / 2) * w_img)
        y1 = int((y_c - h / 2) * h_img)
        x2 = int((x_c + w / 2) * w_img)
        y2 = int((y_c + h / 2) * h_img)

        class_name = COCO_CLASSES[int(row["class_id"])] if int(row["class_id"]) < len(COCO_CLASSES) else "unknown"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_name, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


sample_images = valid["image"].drop_duplicates().sample(min(6, valid["image"].nunique()), random_state=42).tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, img_path in zip(axes.flatten(), sample_images):
    boxes = valid[valid["image"] == img_path]
    vis_img = draw_boxes(img_path, boxes)
    ax.imshow(vis_img)
    ax.set_title(os.path.basename(img_path), fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("coco128_sample_boxes.png")
plt.close()
print("Saved coco128_sample_boxes.png")

print("\nDone. Check coco128_class_distribution.png and coco128_sample_boxes.png")