from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import torch
from model import PollutionModel
from torchvision import transforms
from PIL import Image
import io
import os
from datetime import datetime

app = FastAPI()

# =======================
# Configuração do modelo
# =======================
CLASS_NAMES = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Hazardous"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PollutionModel(NUM_CLASSES)
model.load_state_dict(torch.load("pollution_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =======================
# Endpoints
# =======================
@app.get("/")
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/upload")
async def upload(
        pm25: float = Form(...),
        co: float = Form(...),
        co2: float = Form(...),
        mode: str = Form(...),
        image: UploadFile = File(...)
):
    # processar imagem
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # processar sensores
    if mode == "multimodal":
        sensor_data = torch.tensor([[pm25, co, co2]], dtype=torch.float32).to(DEVICE)
    else:
        sensor_data = torch.zeros((1, 3), dtype=torch.float32).to(DEVICE)

    # forward
    with torch.no_grad():
        output = model(img_tensor, sensor_data)
        pred = torch.argmax(output, dim=1).item()
        label = CLASS_NAMES[pred]

    # guardar imagem como "latest.jpg"
    img.save("latest.jpg")

    # resposta JSON (sem sensores, só relatório)
    report = {
        "prediction_id": pred,
        "prediction_label": label,
        "timestamp": datetime.now().isoformat(),
        "image_url": "/image.jpg"
    }

    return JSONResponse(report)

@app.get("/image.jpg")
async def get_image():
    if os.path.exists("latest.jpg"):
        return FileResponse("latest.jpg", media_type="image/jpeg")
    return JSONResponse({"error": "Nenhuma imagem disponível"})

@app.get("/latest")
async def latest_report():
    if os.path.exists("latest.jpg"):
        return JSONResponse({
            "message": "Último relatório disponível",
            "timestamp": datetime.now().isoformat()
        })
    return JSONResponse({"error": "Nenhum relatório disponível"})
