"""
Fine-tune ResNet18 on Cats vs Dogs
-----------------------------------
Builds directly on cv_dataloader_pipeline.py (must be in the same folder).

Steps:
1. Load pretrained ResNet18
2. Freeze early layers, replace final classifier layer for our 2 classes
3. Train with train_loader, validate each epoch with val_loader
4. Save the best model (based on val accuracy)
5. Final evaluation on test_loader — accuracy + confusion matrix

Usage:
- pip install torch torchvision scikit-learn matplotlib seaborn
- Run after cv_dataloader_pipeline.py's folder structure is ready
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataloader_pipeline import train_loader, val_loader, test_loader, class_to_idx

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(class_to_idx)
EPOCHS = 10
LR = 1e-4
MODEL_SAVE_PATH = "best_resnet18_catsdogs.pt"

print(f"Using device: {DEVICE}")
print(f"Classes: {class_to_idx}")

# ---------------------------------------------------------
# 1. Load pretrained ResNet18
# ---------------------------------------------------------

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers first (feature extractor stays as-is)
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier layer — this one we DO train
# ResNet18's final layer is called "fc", originally outputs 1000 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)  # only this layer is trainable by default

model = model.to(DEVICE)

# ---------------------------------------------------------
# 2. Loss + optimizer
#    Only model.fc.parameters() require grad, so optimizer only updates those
# ---------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ---------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------

def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  -> New best model saved (val acc: {val_acc:.4f})")

print(f"\nTraining done. Best val accuracy: {best_val_acc:.4f}")

# ---------------------------------------------------------
# 4. Final evaluation on TEST set (only once, using best saved model)
# ---------------------------------------------------------

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\nTest accuracy: {test_acc:.4f}")

idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]

print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Saved confusion_matrix.png")