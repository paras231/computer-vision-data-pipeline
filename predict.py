"""
Run inference on a single image using the fine-tuned ResNet18 model.
"""

import sys
import cv2
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

MODEL_PATH = "best_resnet18_catsdogs.pt"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Must match the class_to_idx mapping used during training
idx_to_class = {0: "cats", 1: "dogs"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Rebuild model architecture (must match training script exactly)
# ---------------------------------------------------------

model = models.resnet18(weights=None)  # no need to redownload pretrained weights
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------------------------------------------------
# Preprocessing — same as eval_transform used in training
# ---------------------------------------------------------

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

def predict(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = transform(image=image)["image"]
    input_tensor = transformed.unsqueeze(0).to(DEVICE)  # add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    pred_class = idx_to_class[pred_idx]
    confidence = probs[pred_idx].item()

    print(f"\nImage: {image_path}")
    print(f"Prediction: {pred_class}  (confidence: {confidence:.2%})")
    print("Full probabilities:")
    for idx, cls in idx_to_class.items():
        print(f"  {cls}: {probs[idx].item():.2%}")

    return pred_class, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    predict("download.webp")