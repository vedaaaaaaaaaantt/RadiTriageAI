import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# -------------------
# CONFIG
# -------------------
DATA_DIR = "./"
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 14 diseases
LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

# -------------------
# DATASET
# -------------------
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Keep only required columns
        self.df = self.df[["Path"] + LABELS]
        self.df = self.df.fillna(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Fix path by removing CheXpert-v1.0-small/ prefix
        fixed_path = row["Path"].replace("CheXpert-v1.0-small/", "")
        img_path = os.path.join(self.root_dir, fixed_path)

        img = Image.open(img_path).convert("RGB")

        labels_np = row[LABELS].values.astype("float32")

        # Replace uncertain labels (-1) with 0 
        labels_np[labels_np == -1] = 0

        labels = torch.tensor(labels_np)
        if self.transform:
            img = self.transform(img)

        return img, labels


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

train_dataset = CheXpertDataset(
    "train.csv",
    "./",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_dataset = CheXpertDataset(
    "valid.csv",
    "./",
    transform=transform
)

valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------
# MODEL
# -------------------
model = models.resnet50(weights="IMAGENET1K_V1")

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(LABELS))

model = model.to(DEVICE)

import os

if os.path.exists("models/chexpert_model.pth"):
    print("Loading previous checkpoint...")

    state_dict = torch.load("models/chexpert_model.pth", map_location=DEVICE)

    # Remove final layer weights from old model
    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)

    model.load_state_dict(state_dict, strict=False)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# -------------------
# TRAIN LOOP
# -------------------
# -------------------
# TRAIN LOOP
# -------------------

print("Starting training...")

for epoch in range(10):  # small epochs for Mac
    model.train()
    total_loss = 0

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(train_loader):.4f}")
    print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(train_loader):.4f}")

# Save model after every epoch
torch.save(model.state_dict(), f"models/chexpert_epoch_{epoch+1}.pth")
print(f"Model saved for epoch {epoch+1}")


# -------------------
# VALIDATION
# -------------------

model.eval()
val_loss = 0

with torch.no_grad():
    for images, targets in valid_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, targets)

        val_loss += loss.item()

print(f"Validation Loss: {val_loss/len(valid_loader):.4f}")

torch.save(model.state_dict(), "models/chexpert_model.pth")
print("Model saved successfully.")

