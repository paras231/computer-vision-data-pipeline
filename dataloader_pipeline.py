#  CV Data pipeline :  Dataset + Dataloader

import os
import cv2
import numpy as np
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import  train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATA_DIR = "data"
IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42


# ImageNet normalization stats — standard when using pretrained backbones
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
 
#  Build a dataframe and of image paths + labels


def build_dataframe(dir):
    records = []
    classes = sorted([d for d   in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))])
    class_to_idx = {cls : i for i,cls in enumerate(classes)}
    try:
        for cls in classes:
            cls_dir = os.path.join(dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    records.append({
                        "path": img_path,
                        "label": class_to_idx[cls]
                    })
        df = pd.DataFrame(records)
        return df, class_to_idx
    except Exception as e:
        print(f"Error building dataframe: {e}")
        return pd.DataFrame(), {}

df, class_to_idx = build_dataframe(DATA_DIR)

print(f"""Number of images: {len(df)}""")
print(f"""Classes: {list(class_to_idx.keys())}""")
print(df["label"].value_counts())


# ---------------------------------------------------------
# 2. Stratified split — train / val / test, no leakage
#    (stratify keeps class balance consistent across splits)
# ---------------------------------------------------------
 

train_df,temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)

val_df,test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("Train class balance:\n", train_df["label"].value_counts(normalize=True))
print("Val class balance:\n", val_df["label"].value_counts(normalize=True))
print("Test class balance:\n", test_df["label"].value_counts(normalize=True))



#  ---------------------------------------------------------
# 3. Albumentations pipelines
#    Train gets augmentation, val/test do NOT (must reflect real conditions)
# ---------------------------------------------------------
 
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])
 
eval_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])
 
# ---------------------------------------------------------
# 4. Custom Dataset class
# ---------------------------------------------------------
 
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.paths = dataframe["path"].tolist()
        self.labels = dataframe["label"].tolist()
        self.transform = transform
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
 
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for formats OpenCV doesn't support (e.g. AVIF, WebP variants)
            image = np.array(Image.open(img_path).convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
 
        return image, torch.tensor(label, dtype=torch.long)
 
# ---------------------------------------------------------
# 5. Build datasets + dataloaders
# ---------------------------------------------------------
 
train_dataset = ImageDataset(train_df, transform=train_transform)
val_dataset = ImageDataset(val_df, transform=eval_transform)
test_dataset = ImageDataset(test_df, transform=eval_transform)
 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
 
# ---------------------------------------------------------
# 6. Sanity check — pull one batch and print shapes
# ---------------------------------------------------------
 
if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print(f"\nBatch image tensor shape: {images.shape}")  # [B, C, H, W]
    print(f"Batch label tensor shape: {labels.shape}")
    print(f"Sample labels: {labels[:8].tolist()}")
    print("\nPipeline ready. train_loader / val_loader / test_loader can now be passed into a training loop.")