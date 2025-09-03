# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
import torch
from model import PollutionModel
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# =======================
# Configurações do modelo
# =======================
NUM_CLASSES = 5  # ajuste para o número de classes que treinou
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Criar modelo e carregar pesos
model = PollutionModel(NUM_CLASSES)
model.load_state_dict(torch.load("pollution_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =======================
# Transforms (igual treino)
# =======================
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
    # processar sensores
    sensor_data = torch.tensor([[pm25, co, co2]], dtype=torch.float32).to(DEVICE)

    # processar imagem usando transforms
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # adiciona batch dim

    # forward
    with torch.no_grad():
        output = model(img_tensor, sensor_data)
        pred = torch.argmax(output, dim=1).item()

    return JSONResponse({"prediction": int(pred), "mode": mode})
