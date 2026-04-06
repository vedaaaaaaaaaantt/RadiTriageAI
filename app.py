from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import uuid, io, os

# ---------------- APP ----------------
app = FastAPI(title="AI Radiology Triage")

templates = Jinja2Templates(directory="templates")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- DEVICE ----------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------- MODEL ----------------
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 14)

model.load_state_dict(torch.load("models/chexpert_epoch_10.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- CLASS NAMES ----------------
CLASS_NAMES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices"
]

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- PROCESS IMAGE ----------------
def process_image(image_bytes):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs)[0]

    # Top 3 predictions
    top3 = torch.topk(probs, 3)

    predictions = []
    for i in range(3):
        idx = top3.indices[i].item()
        conf = top3.values[i].item()
        predictions.append((CLASS_NAMES[idx], round(conf, 3)))

    # Risk score
    risk = probs.max().item()

    # Urgency logic
    if risk > 0.85:
        urgency = "IMMEDIATE"
    elif risk > 0.5:
        urgency = "URGENT"
    else:
        urgency = "ROUTINE"

    # Dummy heatmap (safe for now)
    img_np = np.array(pil_img.resize((224, 224)))
    heatmap = np.zeros_like(img_np)
    overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)

    filename = f"{uuid.uuid4().hex[:8]}.jpg"
    path = f"static/{filename}"
    cv2.imwrite(path, overlay)

    return predictions, path, round(risk, 3), urgency

# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("ui.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    predictions, image_path = process_image(image_bytes)

    return {
        "predictions": predictions,
        "image_path": image_path,
        "risk_score": 0.9,   # temp (you can improve later)
        "urgency": "HIGH"
    }
