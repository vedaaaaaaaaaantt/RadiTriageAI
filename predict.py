import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# -------------------
# CONFIG
# -------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

MODEL_PATH = "models/chexpert_model.pth"

# -------------------
# LOAD MODEL
# -------------------
model = models.densenet121(weights=None)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(LABELS))

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# -------------------
# IMAGE TRANSFORM
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------
# LOAD IMAGE FROM COMMAND LINE
# -------------------
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# -------------------
# PREDICTION
# -------------------
with torch.no_grad():
    outputs = model(image)
    probs = torch.sigmoid(outputs)[0]

threshold = 0.5

print("\nPrediction Results:\n")

for i, label in enumerate(LABELS):
    prob = probs[i].item()
    status = "Positive" if prob > threshold else "Negative"
    print(f"{label}: {prob:.4f}  →  {status}")
