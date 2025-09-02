# model_utils.py
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from dataset_and_model import PollutionModel, label_encoder, scaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(label_encoder.classes_)
model = PollutionModel(num_classes)
model.load_state_dict(torch.load("pollution_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path, sensors=None):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    if sensors is None:
        sensors = np.zeros((1, 3), dtype=np.float32)  # se não houver sensores
    else:
        sensors = scaler.transform([sensors])
    sensors = torch.tensor(sensors, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(image, sensors)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return {
        "pred_class": label_encoder.classes_[pred],
        "probs": {cls: float(probs[0, i]) for i, cls in enumerate(label_encoder.classes_)}
    }
