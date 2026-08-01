"""
Gun (YOLO format) - EDA and Bounding Box Visualization
-------------------------------------------------------
Reads YOLO-format label .txt files, draws boxes on images, and reports
basic dataset stats for the gun_annotated dataset.

YOLO label format (one line per object in each .txt file):
    class_id x_center y_center width height
    (all values normalized between 0 and 1, relative to image size)
"""

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

IMAGES_DIR = "gun_annotated/images"
LABELS_DIR = "gun_annotated/labels"

GUN_CLASSES = ["gun"]

# ---------------------------------------------------------
# 1. Parse all label files into a dataframe (one row per object)
# ---------------------------------------------------------

def parse_labels(images_dir, labels_dir):
    records = []
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    for lf in label_files:
        img_name = os.path.splitext(lf)[0]
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
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
# 2. Class distribution (per object)
# ---------------------------------------------------------

valid = df.dropna(subset=["class_id"])
valid = valid.copy()
valid["class_name"] = valid["class_id"].apply(
    lambda i: GUN_CLASSES[i] if i < len(GUN_CLASSES) else f"unknown_{i}"
)

class_counts = valid["class_name"].value_counts()
print("\nObject counts per class:")
print(class_counts)

class_counts.plot(kind="bar", figsize=(8, 4), title="Object count per class (Gun dataset)")
plt.tight_layout()
plt.savefig("gun_class_distribution.png")
plt.close()
print("Saved gun_class_distribution.png")

empty_images = df[df["class_id"].isna()]["image"].nunique()
print(f"\nImages with no labeled objects: {empty_images}")

objects_per_image = valid.groupby("image").size()
print(f"\nObjects per image - min: {objects_per_image.min()}, "
      f"max: {objects_per_image.max()}, mean: {objects_per_image.mean():.2f}")

# ---------------------------------------------------------
# 3. Bounding box size distribution
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
valid["width"].hist(bins=20, ax=axes[0])
axes[0].set_title("Normalized box width distribution")
axes[0].set_xlabel("width")
valid["height"].hist(bins=20, ax=axes[1])
axes[1].set_title("Normalized box height distribution")
axes[1].set_xlabel("height")
plt.tight_layout()
plt.savefig("gun_bbox_size_distribution.png")
plt.close()
print("Saved gun_bbox_size_distribution.png")

# ---------------------------------------------------------
# 4. Draw bounding boxes on sample images to visually verify
# ---------------------------------------------------------

def draw_boxes(image_path, boxes_df):
    img = cv2.imread(image_path)
    if img is None:
        # webp fallback via matplotlib
        import matplotlib.image as mpimg
        img_rgb = mpimg.imread(image_path)
        img = cv2.cvtColor((img_rgb * 255).astype("uint8") if img_rgb.max() <= 1.0 else img_rgb, cv2.COLOR_RGB2BGR)
    h_img, w_img = img.shape[:2]

    for _, row in boxes_df.iterrows():
        if pd.isna(row["class_id"]):
            continue
        x_c, y_c, w, h = row["x_center"], row["y_center"], row["width"], row["height"]
        x1 = int((x_c - w / 2) * w_img)
        y1 = int((y_c - h / 2) * h_img)
        x2 = int((x_c + w / 2) * w_img)
        y2 = int((y_c + h / 2) * h_img)

        class_name = GUN_CLASSES[int(row["class_id"])] if int(row["class_id"]) < len(GUN_CLASSES) else "unknown"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_name, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


n_samples = min(6, valid["image"].nunique())
sample_images = valid["image"].drop_duplicates().sample(n_samples, random_state=42).tolist()

cols = 3
rows = (n_samples + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
for ax, img_path in zip(axes.flatten(), sample_images):
    boxes = valid[valid["image"] == img_path]
    vis_img = draw_boxes(img_path, boxes)
    ax.imshow(vis_img)
    ax.set_title(os.path.basename(img_path), fontsize=8)
    ax.axis("off")

# hide any unused subplots
for ax in axes.flatten()[n_samples:]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("gun_sample_boxes.png")
plt.close()
print("Saved gun_sample_boxes.png")

print("\nDone. Check gun_class_distribution.png, gun_bbox_size_distribution.png, and gun_sample_boxes.png")
