#executar: python3 evaluate_multimodal.py

# evaluate_multimodal.py
import os, json, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Paths
OUT = "output_model"
CSV = "registos_labeled.csv"
IMG_DIR = "imagens"

# carrega artefactos
with open(os.path.join(OUT, "label_map.json")) as f:
    label_map = json.load(f)
idx2label = {int(k):v for k,v in label_map["idx2label"].items()}

with open(os.path.join(OUT, "scaler.pkl"), "rb") as f:
    scdata = pickle.load(f)
scaler = scdata["scaler"]
sensor_cols = scdata["sensor_cols"]

# lê csv e filtra imagens existentes
df = pd.read_csv(CSV)
df["image_path"] = df["image"].apply(lambda x: os.path.join(IMG_DIR, x))
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

# split similar ao treino (usar mesmo seed e proporções)
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val_df, test_df  = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

# transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# modelo (mesma arquitetura)
class MultimodalNet(torch.nn.Module):
    def __init__(self, num_sensor_feats, num_classes):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=True)
        backbone.classifier = torch.nn.Identity()
        self.image_model = backbone
        self.sensor_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_sensor_feats, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU()
        )
        combined = backbone.last_channel + 32
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(combined, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )
    def forward(self, image, sensors):
        img_feat = self.image_model(image)
        sensor_feat = self.sensor_mlp(sensors)
        x = torch.cat([img_feat, sensor_feat], dim=1)
        return self.classifier(x)

model = MultimodalNet(num_sensor_feats=len(sensor_cols), num_classes=len(idx2label)).eval()
model.load_state_dict(torch.load(os.path.join(OUT, "multimodal_best.pth"), map_location="cpu"))

# inferir
y_true = []
y_pred = []

for _, row in test_df.iterrows():
    img = Image.open(row["image_path"]).convert("RGB")
    x = transform(img).unsqueeze(0)
    sensors = row[sensor_cols].fillna(0).values.reshape(1,-1).astype(float)
    sensors = scaler.transform(sensors).astype(np.float32)
    s = torch.from_numpy(sensors)
    with torch.no_grad():
        out = model(x, s)
        pred = out.argmax(dim=1).item()
    y_true.append(row["label"])
    y_pred.append(idx2label[pred])

print("Classification report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred, labels=list(sorted(set(y_true))))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)))
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()
