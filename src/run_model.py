import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F

# Load DenseNet-121 (CheXNet-style backbone)
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
img = Image.open("data/sample_xray.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(img)
    probs = F.softmax(output, dim=1)

# Risk score (simple proxy)
risk_score = probs.max().item()

# Urgency logic (YOUR innovation)
if risk_score > 0.85:
    urgency = "🔴 IMMEDIATE"
elif risk_score > 0.50:
    urgency = "🟠 URGENT"
else:
    urgency = "🟢 ROUTINE"

print(f"Risk score: {risk_score:.3f}")
print(f"Urgency level: {urgency}")

