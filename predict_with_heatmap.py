import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import sys
import os

# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device("cpu")

LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

# -------------------------
# LOAD MODEL (ResNet50)
# -------------------------
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))
model.load_state_dict(torch.load("models/chexpert_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------
# IMAGE
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_path = sys.argv[1]
original_image = Image.open(img_path).convert("RGB")
input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

# -------------------------
# GRAD-CAM
# -------------------------
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer = model.layer4
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Forward
output = model(input_tensor)
probs = torch.sigmoid(output).detach().cpu().numpy()[0]

print("\nPrediction Results:\n")
for label, prob in zip(LABELS, probs):
    print(f"{label}: {prob:.4f}")

top_class = np.argmax(probs)
print(f"\nTop Prediction: {LABELS[top_class]}")

# Backward
model.zero_grad()
output[0, top_class].backward()

# -------------------------
# HEATMAP
# -------------------------
feature_map = features[0]
grads = gradients[0]

pooled_grads = torch.mean(grads, dim=[0, 2, 3])

for i in range(feature_map.shape[1]):
    feature_map[:, i, :, :] *= pooled_grads[i]

heatmap = torch.mean(feature_map, dim=1).squeeze()
heatmap = heatmap.detach().cpu().numpy()

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap) + 1e-8

heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
superimposed_img = cv2.addWeighted(original_cv, 0.6, heatmap, 0.4, 0)

os.makedirs("heatmaps", exist_ok=True)
output_path = "heatmaps/output_heatmap.jpg"
cv2.imwrite(output_path, superimposed_img)

print(f"\nHeatmap saved at: {output_path}")
