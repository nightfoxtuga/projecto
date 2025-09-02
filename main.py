import os
import time
import threading
import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import numpy as np
import joblib

# =======================
# CONFIGURAÇÕES
# =======================
MODEL_PATH = "pollution_model.pth"
SCALER_PATH = "scaler.pkl"
IMG_PATH = "static/image.jpg"
CAPTURE_INTERVAL = 5

# Classes reais (iguais às do treino!)
class_names = ["Good", "Moderate", "Unhealthy"]

# =======================
# MODELO
# =======================
class PollutionModel(nn.Module):
    def __init__(self, num_classes, sensor_input_dim=3):
        super(PollutionModel, self).__init__()
        self.cnn = models.resnet18(pretrained=False)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.sensor_fc = nn.Sequential(
            nn.Linear(sensor_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, sensors):
        img_feat = self.cnn(img)
        sensor_feat = self.sensor_fc(sensors)
        combined = torch.cat((img_feat, sensor_feat), dim=1)
        out = self.fc(combined)
        return out

# =======================
# CARREGAR MODELO E SCALER
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PollutionModel(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Scaler para normalizar sensores
scaler = joblib.load(SCALER_PATH)

# Transform da imagem
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =======================
# INFERÊNCIA
# =======================
def run_inference(img_path, sensors, mode="multimodal"):
    """
    img_path: caminho da imagem
    sensors: lista com [pm25, co, co2] brutos (não normalizados)
    mode: "image_only" ou "multimodal"
    """
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    if mode == "multimodal":
        sensors_norm = scaler.transform([sensors])[0]  # normalizar
    else:
        sensors_norm = [0, 0, 0]  # "imagem apenas" -> ignora sensores

    sensors_tensor = torch.tensor(sensors_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, sensors_tensor)
        _, predicted = torch.max(outputs, 1)

    result = {
        "prediction": class_names[predicted.item()],
        "mode": mode,
        "sensors_used": sensors if mode == "multimodal" else None
    }
    return result

# =======================
# FASTAPI
# =======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

capture_thread = None
running = False
latest_report = {}

# Dummy sensor read (substitui isto por leitura real do Arduino)
def read_sensors():
    pm25 = np.random.uniform(5, 60)
    co = np.random.uniform(0.1, 5)
    co2 = np.random.uniform(400, 2000)
    return [pm25, co, co2]

def capture_loop(interval, mode):
    global running, latest_report
    while running:
        # Capturar imagem (simulação: usa sempre a mesma)
        img_path = IMG_PATH

        if mode == "multimodal":
            sensors = read_sensors()
        else:
            sensors = [0, 0, 0]

        result = run_inference(img_path, sensors, mode)
        latest_report = {"report": result}
        time.sleep(interval)

@app.post("/start_capture")
async def start_capture(request: Request):
    global running, capture_thread
    if running:
        return {"status": "already_running"}
    body = await request.json()
    interval = body.get("interval", CAPTURE_INTERVAL)
    mode = body.get("mode", "multimodal")
    running = True
    capture_thread = threading.Thread(target=capture_loop, args=(interval, mode), daemon=True)
    capture_thread.start()
    return {"status": "started", "mode": mode}

@app.post("/stop_capture")
async def stop_capture():
    global running
    running = False
    return {"status": "stopped"}

@app.get("/latest")
async def get_latest():
    return latest_report if latest_report else {"report": None}

@app.get("/image.jpg")
async def get_image():
    return FileResponse(IMG_PATH)

@app.post("/capture")
async def capture_once(request: Request):
    """Captura e infere apenas uma vez"""
    body = await request.json()
    mode = body.get("mode", "multimodal")

    if mode == "multimodal":
        sensors = read_sensors()
    else:
        sensors = [0, 0, 0]

    result = run_inference(IMG_PATH, sensors, mode)
    return JSONResponse(content={"report": result})
