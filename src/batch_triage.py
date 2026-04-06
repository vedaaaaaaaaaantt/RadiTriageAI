import os
import uuid
import csv
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_urgency(score):
    if score > 0.85:
        return "IMMEDIATE"
    elif score > 0.50:
        return "URGENT"
    else:
        return "ROUTINE"

image_folder = "data/batch"
results = []

for fname in os.listdir(image_folder):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    case_id = str(uuid.uuid4())[:8]   # short case ID
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    img_path = os.path.join(image_folder, fname)
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        risk = probs.max().item()

    urgency = get_urgency(risk)

    results.append({
        "case_id": case_id,
        "time": timestamp,
        "image": fname,
        "risk": round(risk, 3),
        "urgency": urgency
    })

# Sort by highest risk first
results.sort(key=lambda x: x["risk"], reverse=True)

# Print worklist
print("\n=== RADIOLOGY WORKLIST ===")
for r in results:
    print(f"[{r['case_id']}] {r['time']} | {r['urgency']} | score={r['risk']} | {r['image']}")

# Save CSV (hospital-style output)
os.makedirs("output", exist_ok=True)
with open("output/worklist.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["case_id", "time", "image", "risk", "urgency"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nWorklist saved to output/worklist.csv")

