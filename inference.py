import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import serial  # se quiseres ler do Arduino via serial
from dataset_and_model import PollutionModel, label_encoder, scaler  # importar os mesmos objetos usados no treino

# ==========================
# Configurações
# ==========================
IMG_PATH = "captured.jpg"   # imagem capturada pela PiCam
MODEL_PATH = "pollution_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Carregar modelo
# ==========================
num_classes = len(label_encoder.classes_)
model = PollutionModel(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==========================
# Preprocessamento
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

# ==========================
# Função inferência
# ==========================
def predict(image_tensor, sensors=None):
    if sensors is None:
        # Sem sensores → usar vetor nulo
        sensors = np.zeros((1, 3), dtype=np.float32)
    else:
        sensors = scaler.transform([sensors])  # normalizar
    sensors_tensor = torch.tensor(sensors, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor, sensors_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    return label_encoder.inverse_transform([pred])[0], probs.cpu().numpy()

# ==========================
# Exemplo de uso
# ==========================
if __name__ == "__main__":
    # Carregar imagem
    img_tensor = preprocess_image(IMG_PATH)

    # Modo só imagem
    classe, probs = predict(img_tensor)
    print(f"Previsão só com imagem: {classe}, Probabilidades: {probs}")

    # Modo multimodal (exemplo com sensores lidos do Arduino)
    # Aqui suponho que já tens os valores como floats
    sensor_values = [25.0, 0.8, 450.0]  # pm25, co, co2
    classe_mm, probs_mm = predict(img_tensor, sensor_values)
    print(f"Previsão multimodal: {classe_mm}, Probabilidades: {probs_mm}")
