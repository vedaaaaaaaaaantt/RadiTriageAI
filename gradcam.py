import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn.functional as F

# -------------------------
# CONFIG
# -------------------------
IMAGE_PATH = "valid/patient64541/study1/view1_frontal.jpg"
DEVICE = torch.device("cpu")

# CheXpert labels (14 classes)
CLASS_NAMES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices"
]

# -------------------------
# LOAD MODEL
# -------------------------
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 14)

model.load_state_dict(torch.load("models/chexpert_epoch_10.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------
# TARGET LAYER
# -------------------------
target_layer = model.layer4[-1]

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# LOAD IMAGE
# -------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# -------------------------
# PREDICTION
# -------------------------
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.sigmoid(outputs)[0]  # multi-label

# Get top 3 predictions
top3 = torch.topk(probs, 3)

print("\n🔍 Top Predictions:")
for i in range(3):
    class_idx = top3.indices[i].item()
    confidence = top3.values[i].item()
    print(f"{CLASS_NAMES[class_idx]}: {confidence:.2f}")

# -------------------------
# GRAD-CAM (for top class)
# -------------------------
target_class = top3.indices[0].item()

cam = GradCAM(model=model, target_layers=[target_layer])

grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

# -------------------------
# OVERLAY
# -------------------------
img_np = np.array(img.resize((224, 224))) / 255.0
visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

# -------------------------
# SAVE OUTPUT
# -------------------------
output_path = "gradcam_output.jpg"
cv2.imwrite(output_path, visualization)

print(f"\n✅ Grad-CAM saved as {output_path}")
