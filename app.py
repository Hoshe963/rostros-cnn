from flask import Flask, render_template, request
from PIL import Image
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

# -------------------------
# CONFIGURACIÓN BÁSICA
# -------------------------
IMG_SIZE = 128  # Debe coincidir con tu entrenamiento

# Misma arquitectura que tu CNNet en Colab
class CNNet(nn.Module):
    def __init__(self, num_classes):
        super(CNNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.fc_input_features = 32 * (IMG_SIZE // 4) * (IMG_SIZE // 4)

        self.fc1 = nn.Linear(self.fc_input_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------
# ETIQUETAS (clases)
# -------------------------
labels = [
    "Aaron Lopez",
    "Jerika Anthonella",
    "Mark Landeo",
    "Miguel Adrian"
]
num_classes = len(labels)

# -------------------------
# TRANSFORMACIONES
# -------------------------
img_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------
# CARGAR MODELO
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNet(num_classes=num_classes)
model.load_state_dict(torch.load("simplenet_weights.pth", map_location=device))
model.to(device)
model.eval()

# -------------------------
# FUNCIÓN DE PREDICCIÓN
# -------------------------
def predict_image(pil_image):
    tensor = img_transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probabilities, 0)

    predicted_label = labels[pred_idx.item()]
    confidence = conf.item() * 100
    return predicted_label, confidence

# -------------------------
# APLICACIÓN FLASK
# -------------------------
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != "":
                image_bytes = file.read()
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                prediction, confidence = predict_image(pil_image)

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
