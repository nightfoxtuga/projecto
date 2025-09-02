#executar: python3 infer_multimodal.py --img imagens/img_010.jpg --sensors 345 120 12 35 40 25.3 45.2 1012.3 23810 27.0 40.0 22.1

# infer_multimodal.py
import os, json, pickle, argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="output_model/multimodal_traced.pt")
parser.add_argument("--scaler", default="output_model/scaler.pkl")
parser.add_argument("--labelmap", default="output_model/label_map.json")
parser.add_argument("--img", required=True, help="imagem a avaliar")
parser.add_argument("--sensors", nargs="+", type=float, help="valores sensores na ordem do scaler (opcional)")
parser.add_argument("--csv_line", help="opcional: procura linha no csv (formato: image,...)")
args = parser.parse_args()

# carrega artefactos
with open(args.labelmap) as f:
    label_map = json.load(f)
idx2label = {int(k):v for k,v in label_map["idx2label"].items()}

with open(args.scaler, "rb") as f:
    sc = pickle.load(f)
scaler = sc["scaler"]
sensor_cols = sc["sensor_cols"]

# transform imagem
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open(args.img).convert("RGB")
x = transform(img).unsqueeze(0)

# sensors: se passagem via --sensors usa essa ordem. Se não, tenta procurar csv_line (não implementado aqui).
if args.sensors:
    sensors = np.array(args.sensors, dtype=float).reshape(1,-1)
    if sensors.shape[1] != len(sensor_cols):
        raise SystemExit(f"Esperado {len(sensor_cols)} sensores, recebido {sensors.shape[1]}")
else:
    # default zeros
    sensors = np.zeros((1, len(sensor_cols)), dtype=float)

sensors_scaled = scaler.transform(sensors).astype(np.float32)
s = torch.from_numpy(sensors_scaled)

# carrega modelo TorchScript
model = torch.jit.load(args.model, map_location="cpu")
model.eval()

with torch.no_grad():
    out = model(x, s)  # TorchScript espera (image_tensor, sensor_tensor)
    probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    pred = int(out.argmax(dim=1).item())

print("Prediction:", idx2label[pred])
for i,lab in idx2label.items():
    print(f"  {lab}: {probs[int(i)]:.3f}")
