# Weapon Detection EDA pipeline
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import albumentations as A

# List images

TRAIN_DATA_IMAGES = "weapon_data2/weapon_detection/train/images"
VAL_DATA_IMAGES = "weapon_data2/weapon_detection/val/images"

def list_images(dir):
    image_ext = ('.jpg','.jpeg','.png','.bmp','.tiff','.gif','.webp')
    paths = []
    if os.path.exists(dir):
        for root,_,files in os.walk(dir):
            # print(files)
            for f in files:
                if f.lower().endswith(image_ext):
                    paths.append(os.path.join(root,f))
    return paths


paths = list_images(TRAIN_DATA_IMAGES)

# print(paths)

# build data frame

def build_dataframe():
    records = []
    try:
        for p in paths:
            with Image.open(p) as img:
                w,h = img.size
                fmt = img.format
                label = os.path.basename(os.path.dirname(p))
            records.append({
                "width":w,
                "height":h,
                "format":fmt,
                "label":label,
                "path":p
            })
        return records
    except Exception as e:
        print(f"Error processing {p}: {e}")


records = build_dataframe()

# print(len(records))

df = pd.DataFrame(records)

print(df.head())
print(df["format"].value_counts())
    


# Perform augmentation and test it if applied properly (on a single image)

def perform_augmentation():
    if paths:
        img = None
        sample_img_path = None
        for p in paths:
            raw = cv2.imread(p)
            if raw is not None:
                img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                sample_img_path = p
                break
        if img is None:
            print("No readable images found for augmentation (OpenCV could not decode any file).")
        if img is not None:
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
            plt.savefig("weapon_agumneted_example.png")
            plt.close()
            # print("\nSaved augmentation_example.png")




perform_augmentation()
    