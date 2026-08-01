import os
from PIL import Image
import pandas as pd
import hashlib
import cv2
import matplotlib.pyplot as plt
dir = "data"

def list_images(path):
  exts = (".jpg",".jpeg",".avif","webp",".png",".bnp")
  paths = []
  for root ,_ , files in os.walk(path):
    for f in files:
      if f.lower().endswith(exts):
        paths.append(os.path.join(root,f))
  return paths

# print(list_images(dir)


img_paths = list_images(dir)

# print(img_paths)

#  EDA - Exploratory data analysis -> sizes , formats , distribution 


def file_hash(path):
    with open(path,"rb") as f:
        return hashlib.md5(f.read()).hexdigest()


records = []


for p in img_paths:
    try:
        with Image.open(p) as img:
            w,h = img.size
            fmt = img.format
            label = os.path.basename(os.path.dirname(p))
            hash = file_hash(p)
        records.append({
        "path":p,
        "width":w,
        "height":h,
        "format":fmt,
        "label":label,
        "hash":hash
        })
    except Exception as e:
        print(f"Error processing {p}: {e}")


# print(records)

df = pd.DataFrame(records)

print(df.head())
    

print(df["format"].value_counts())

if(df["label"].nunique() < 20):
    print(df["label"].value_counts())
  


# ---------------------------------------------------------
# 3. CLEANING CHECKS — duplicates, corrupted files, odd sizes
# ---------------------------------------------------------
 



hashes = {}
duplicates = []
corrupted = []

for p in records:
    try:
        h = file_hash(p["path"])
        if h in hashes:
            duplicates.append((p["path"],hashes[h]))
        else:
            hashes[h] = p
    except Exception as e:
        print(f"Error processing {p}: {e}")
        corrupted.append(p)


# print(f"Number of duplicates found: {len(duplicates)}",duplicates)
# print(f"Number of corrupted files found: {len(corrupted)}",corrupted)

# Flag unusually small or large images (basic outlier check)

if not df.empty:
    median_width = df["width"].median()
    median_height = df["height"].median()
    outliers = df[(df["width"] < median_width * 0.2) | (df["width"] > median_width * 5)]
    print(f"\nSize outliers found: {len(outliers)}")
    if not outliers.empty:
        print(outliers[["path", "width", "height"]])




# ---------------------------------------------------------
# 4. BASIC AUGMENTATION — visualize before/after
# ---------------------------------------------------------
 
try:
    import albumentations as A
 
    if img_paths:
        img = None
        sample_img_path = None
        for candidate in img_paths:
            raw = cv2.imread(candidate)
            if raw is not None:
                img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                sample_img_path = candidate
                break
        if img is None:
            print("No readable images found for augmentation (OpenCV could not decode any file).")
        if img is not None:
            print(sample_img_path)
            transform = A.Compose([
                A.HorizontalFlip(p=1.0),
                A.RandomBrightnessContrast(p=1.0),
                A.Rotate(limit=25, p=1.0),
            ])

            augmented = transform(image=img)["image"]

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(augmented)
            axes[1].set_title("Augmented")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig("augmentation_example.png")
            plt.close()
            print("\nSaved augmentation_example.png")
 
except ImportError:
    print("\nAlbumentations not installed. Run: pip install albumentations")
 
print("\nDone. Check the saved PNGs: class_distribution.png, sample_grid.png, augmentation_example.png")