import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import subprocess

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

# -------------------
# LOAD MODEL
# -------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)  # 5 labels
    model.load_state_dict(torch.load("models/chexpert_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------
# UI
# -------------------
st.title("🩻 AI Radiology Triage System")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)[0]

    st.subheader("Prediction Results")

    for i, label in enumerate(LABELS):
        st.write(f"{label}: {probs[i]:.4f}")

    # Save temporary image
    os.makedirs("temp", exist_ok=True)
    temp_path = "temp/input.jpg"
    image.save(temp_path)

    # Run heatmap script
    subprocess.run(["python", "predict_with_heatmap.py", temp_path])

    if os.path.exists("heatmaps/output_heatmap.jpg"):
        st.subheader("Grad-CAM Heatmap")
        st.image("heatmaps/output_heatmap.jpg", use_column_width=True)
