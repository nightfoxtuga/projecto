from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from datetime import datetime
from PIL import Image
import torch, io, os

from model import PollutionModel
from torchvision import transforms

app = FastAPI()

# =======================
# Modelo de poluição
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
# Estado da captura
# =======================
capture_active = False
capture_mode = "multimodal"
capture_interval = 60  # segundos

LATEST_REPORT_FILE = "latest_report.json"

# =======================
# Endpoints Frontend
# =======================
@app.get("/")
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/latest")
async def latest_report():
    if os.path.exists(LATEST_REPORT_FILE):
        with open(LATEST_REPORT_FILE, "r") as f:
            report = json.load(f)
        return JSONResponse(report)
    return JSONResponse({"error": "Nenhum relatório disponível"})

@app.get("/image.jpg")
async def get_image():
    if os.path.exists("latest.jpg"):
        return FileResponse("latest.jpg", media_type="image/jpeg")
    return JSONResponse({"error": "Nenhuma imagem disponível"})

# =======================
# Endpoints de controle (Frontend envia para backend)
# =======================
@app.post("/start_capture")
async def start_capture(interval: int = Form(60), mode: str = Form("multimodal")):
    global capture_active, capture_mode, capture_interval
    capture_active = True
    capture_mode = mode
    capture_interval = interval
    return JSONResponse({"status": "Captura iniciada", "mode": mode, "interval": interval})

@app.post("/stop_capture")
async def stop_capture():
    global capture_active
    capture_active = False
    return JSONResponse({"status": "Captura parada"})

@app.post("/capture_once")
async def capture_once(mode: str = Form("multimodal"), pm25: float = Form(...),
                       co: float = Form(...), co2: float = Form(...), image: UploadFile = File(...)):
    # processa imagem
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # processa sensores
    if mode == "multimodal":
        sensor_data = torch.tensor([[pm25, co, co2]], dtype=torch.float32).to(DEVICE)
    else:
        sensor_data = torch.zeros((1, 3), dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor, sensor_data)
        pred = torch.argmax(output, dim=1).item()
        label = CLASS_NAMES[pred]

    # salva imagem
    img.save("latest.jpg")

    report = {
        "prediction_id": pred,
        "prediction_label": label,
        "timestamp": datetime.now().isoformat(),
        "image_url": "/image.jpg"
    }
    with open(LATEST_REPORT_FILE, "w") as f:
        json.dump(report, f)
    return JSONResponse(report)

# =======================
# Endpoint que o Raspberry consulta
# =======================
@app.get("/get_config")
async def get_config():
    return JSONResponse({
        "capture_active": capture_active,
        "capture_mode": capture_mode,
        "capture_interval": capture_interval
    })
