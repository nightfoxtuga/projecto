# dataset_and_model.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# =======================
# CONFIGURAÇÕES
# =======================
CSV_PATH = "sensor_data.csv"
IMG_DIR = "images"

# =======================
# 1. Carregar dataset
# =======================
df = pd.read_csv(CSV_PATH)

for col in ["pm25", "co", "co2"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["pm25", "co", "co2", "qa"]).reset_index(drop=True)

# =======================
# 2. Preparar labels
# =======================
label_encoder = LabelEncoder()
df["qa"] = label_encoder.fit_transform(df["qa"])

# =======================
# 3. Normalização
# =======================
scaler = StandardScaler()
df[["pm25", "co", "co2"]] = scaler.fit_transform(df[["pm25", "co", "co2"]])

# =======================
# 4. Split train/val/test
# =======================
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["qa"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["qa"], random_state=42)

# =======================
# 5. Dataset personalizado
# =======================
class PollutionDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.images = self.df["image_name"].values
        self.sensor_data = self.df[["pm25", "co", "co2"]].values.astype(np.float32)
        self.labels = self.df["qa"].values.astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.images[idx]) + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        sensors = torch.tensor(self.sensor_data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, sensors, label

# =======================
# 6. Transforms
# =======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =======================
# 7. Modelo
# =======================
class PollutionModel(nn.Module):
    def __init__(self, num_classes, sensor_input_dim=3):
        super(PollutionModel, self).__init__()

        self.cnn = models.resnet18(pretrained=True)
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
        out = self.fc(combined)
        return out
