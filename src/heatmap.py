import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import os

# Load model
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.eval()

# Hook storage
features = None
gradients = None

def save_features(module, input, output):
    global features
    features = output

def save_gradients(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# Register hooks
model.features.register_forward_hook(save_features)
model.features.register_backward_hook(save_gradients)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
img_path = "data/batch/xray_01.jpg"
orig = cv2.imread(img_path)
orig = cv2.resize(orig, (224, 224))

img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0)

# Forward
output = model(img)
score, idx = torch.max(output, 1)

# Backward
model.zero_grad()
score.backward()

# Grad-CAM
weights = gradients.mean(dim=[2, 3], keepdim=True)
cam = (weights * features).sum(dim=1).squeeze()
cam = torch.relu(cam)
cam = cam.detach().numpy()

cam = cv2.resize(cam, (224, 224))
cam = cam / cam.max()

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

os.makedirs("output", exist_ok=True)
cv2.imwrite("output/heatmap_result.jpg", overlay)

print("Heatmap saved → output/heatmap_result.jpg")

