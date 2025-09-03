# main.py
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pathlib import Path
from datetime import datetime
import io, time
import joblib
import torch
from torchvision import transforms
from PIL import Image

from model import PollutionModel  # assume model.py está no mesmo diretório

app = FastAPI()

# -----------------------
# CONFIG (ajusta caminhos)
# -----------------------
BASE = Path(__file__).resolve().parent
INDEX_FILE = BASE / "index.html"
MODEL_PATH = BASE / "pollution_model.pth"
SCALER_PATH = BASE / "scaler.pkl"   # opcional, se existir será carregado
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)
LATEST_IMAGE = DATA_DIR / "latest.jpg"

# classes na mesma ordem do LabelEncoder usado no treino
CLASS_NAMES = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Hazardous",
]
NUM_CLASSES = len(CLASS_NAMES)

# dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# carregar modelo
# -----------------------
model = PollutionModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# optional scaler (igual ao treino)
scaler = None
if SCALER_PATH.exists():
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        scaler = None

# transforms iguais ao treino
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ultimo relatório (guardado em memória) — este é o que o frontend mostra
latest_report = {}

def make_public_report(pred_id: int, mode: str) -> dict:
    label = CLASS_NAMES[pred_id] if 0 <= pred_id < len(CLASS_NAMES) else f"class_{pred_id}"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": ts,
        "prediction_id": int(pred_id),
        "prediction_label": label,
        "mode": mode,
        "image_url": f"/image.jpg?cache={int(time.time())}"
    }

# -----------------------
# Endpoints
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    # devolve index.html (ficheiro separado)
    if INDEX_FILE.exists():
        return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html não encontrado</h1>", status_code=404)

@app.post("/upload")
async def upload(
        pm25: float = Form(...),
        co: float = Form(...),
        co2: float = Form(...),
        mode: str = Form("multimodal"),
        image: UploadFile = File(...)
):
    """
    Recebe imagem + sensores + mode do Raspberry:
      - se mode == "multimodal": usa sensores (se scaler existir, aplica)
      - se mode == "image_only": ignora sensores (usa zeros)
    Gera relatório (classificação) e guarda latest image/report.
    """
    global latest_report

    # ler imagem
    img_bytes = await image.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(pil).unsqueeze(0).to(DEVICE)  # [1,3,224,224]

    # preparar sensores (apenas usados se multimodal)
    if mode == "multimodal":
        sensors = [[pm25, co, co2]]
        if scaler:
            try:
                sensors = scaler.transform(sensors)
            except Exception:
                pass
        sensor_tensor = torch.tensor(sensors, dtype=torch.float32, device=DEVICE)
    else:
        sensor_tensor = torch.zeros((1, 3), dtype=torch.float32, device=DEVICE)

    # inferência
    with torch.no_grad():
        logits = model(img_tensor, sensor_tensor)
        pred_id = int(torch.argmax(logits, dim=1).item())

    # salvar a imagem atual
    with open(LATEST_IMAGE, "wb") as f:
        f.write(img_bytes)

    # gerar relatório público (sem sensors)
    latest_report = make_public_report(pred_id, mode)
    return JSONResponse({"status": "ok", **latest_report})


@app.get("/latest")
async def get_latest():
    if latest_report:
        return JSONResponse(latest_report)
    return JSONResponse({"status": "no_data"})


@app.get("/image.jpg")
async def image_jpg():
    if LATEST_IMAGE.exists():
        return FileResponse(LATEST_IMAGE, media_type="image/jpeg")
    return JSONResponse({"status": "no_image"})


# stubs para a UI sem quebra (captura/controlos)
@app.post("/capture")
async def capture(request: Request):
    data = await request.json()
    # o frontend pede /capture e espera um objecto com o relatório;
    # devolvemos o último relatório (o Raspberry é quem realmente captura)
    if latest_report:
        return JSONResponse(latest_report)
    return JSONResponse({"status": "no_data"})

@app.post("/start_capture")
async def start_capture(request: Request):
    return JSONResponse({"status": "ok", "message": "simulated start"})

@app.post("/stop_capture")
async def stop_capture():
    return JSONResponse({"status": "ok", "message": "simulated stop"})
