import os
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from PIL import Image

# ======================
# CONFIG
# ======================
MODEL_PATH = "pollution_model.pth"
SCALER_PATH = "scaler.pkl"
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "sensor_data.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")

os.makedirs(IMG_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["pm25", "co", "co2", "image_name"]).to_csv(CSV_PATH, index=False)

# ======================
# FastAPI setup
# ======================
app = FastAPI()
app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")

# ======================
# Modelo
# ======================
class PollutionModel(nn.Module):
    def __init__(self, num_classes, sensor_input_dim=3):
        super(PollutionModel, self).__init__()
        self.cnn = models.resnet18(weights=None)
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
        return self.fc(combined)

# carregar modelo e scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load(SCALER_PATH)
model = PollutionModel(num_classes=5)  # corrigido para bater com o checkpoint
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# transform para imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# Endpoints
# ======================

@app.get("/", response_class=HTMLResponse)
def index():
    # certifique-se que o index.html está no mesmo diretório do main.py
    if not os.path.exists("index.html"):
        return HTMLResponse("<h1>index.html não encontrado</h1>", status_code=404)
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
async def upload_data(
        pm25: float = Form(...),
        co: float = Form(...),
        co2: float = Form(...),
        mode: str = Form("multimodal"),
        image: UploadFile = File(...)
):
    """Recebe dados do Raspberry Pi"""
    img_name = f"img_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = os.path.join(IMG_DIR, img_name)

    # guardar imagem
    with open(img_path, "wb") as f:
        f.write(await image.read())

    # atualizar CSV
    df = pd.read_csv(CSV_PATH)
    new_row = {"pm25": pm25, "co": co, "co2": co2, "image_name": img_name}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    # preparar inputs
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    if mode == "multimodal":
        sensors = scaler.transform([[pm25, co, co2]])
        sensors_tensor = torch.tensor(sensors, dtype=torch.float32).to(device)
    else:
        sensors_tensor = torch.zeros((1, 3), dtype=torch.float32).to(device)

    # inferência
    with torch.no_grad():
        output = model(image, sensors_tensor)
        predicted_class = torch.argmax(output, 1).item()

    return {
        "status": "ok",
        "prediction": int(predicted_class),
        "image_url": f"/images/{img_name}"
    }

@app.get("/latest")
def get_latest():
    """Última entrada"""
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return {"status": "no data"}
    last_row = df.iloc[-1]
    return {
        "pm25": float(last_row["pm25"]),
        "co": float(last_row["co"]),
        "co2": float(last_row["co2"]),
        "image_url": f"/images/{last_row['image_name']}"
    }

@app.get("/image.jpg")
def get_latest_image():
    """Imagem atual"""
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return FileResponse("placeholder.jpg")  # certifique-se de ter uma imagem placeholder
    last_row = df.iloc[-1]
    return FileResponse(os.path.join(IMG_DIR, last_row["image_name"]))
