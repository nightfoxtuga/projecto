from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import shutil
import torch
from model import PollutionModel  # seu modelo
from utils import preprocess_image, preprocess_sensors  # funções de pré-processamento

app = FastAPI()

# Diretório base do projeto (garante que index.html será encontrado)
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

# Carrega o modelo
NUM_CLASSES = 5  # ajusta conforme seu checkpoint
MODEL_PATH = BASE_DIR / "checkpoint.pt"
model = PollutionModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

@app.get("/", response_class=HTMLResponse)
async def index():
    # Lê index.html de forma segura
    return INDEX_FILE.read_text(encoding="utf-8")

@app.post("/upload")
async def upload(
        image: UploadFile = File(...),
        pm25: float = Form(...),
        co: float = Form(...),
        co2: float = Form(...),
        mode: str = Form("multimodal")
):
    # Salva imagem temporariamente
    temp_image_path = BASE_DIR / "temp.jpg"
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Pré-processa
    img_tensor = preprocess_image(temp_image_path)
    sensor_tensor = preprocess_sensors(pm25, co, co2)

    # Preditivo
    with torch.no_grad():
        if mode == "image_only":
            output = model(img_tensor=img_tensor)
        elif mode == "multimodal":
            output = model(img_tensor=img_tensor, sensor_tensor=sensor_tensor)
        else:
            return JSONResponse({"error": "Invalid mode"}, status_code=400)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": prediction}
