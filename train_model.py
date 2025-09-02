import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image

# =======================
# CONFIGURAÇÕES
# =======================
CSV_PATH = "sensor_data.csv"  # ficheiro com dados
IMG_DIR = "../Documents/projetofimcurso/project/images"  # pasta com imagens
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

# =======================
# 1. Carregar dataset
# =======================
df = pd.read_csv(CSV_PATH)

# Forçar conversão das colunas numéricas, substituir inválidos por NaN e remover linhas com NaN
for col in ["pm25", "co", "co2"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["pm25", "co", "co2", "qa"])  # garantir colunas limpas
df = df.reset_index(drop=True)

# =======================
# 2. Preparar labels (qa)
# =======================
label_encoder = LabelEncoder()
df["qa"] = label_encoder.fit_transform(df["qa"])

print("Classes encontradas:", list(label_encoder.classes_))

# =======================
# 3. Normalização dos sensores
# =======================
scaler = StandardScaler()
df[["pm25", "co", "co2"]] = scaler.fit_transform(df[["pm25", "co", "co2"]])

# =======================
# 4. Split train/val/test
# =======================
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["qa"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["qa"], random_state=42)

print("Train classes:", train_df["qa"].unique())
print("Val classes:", val_df["qa"].unique())
print("Test classes:", test_df["qa"].unique())

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
        self.labels = self.df["qa"].values.astype(int)  # já são inteiros

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
# 6. Transforms para imagens
# =======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =======================
# 7. DataLoaders
# =======================
train_dataset = PollutionDataset(train_df, IMG_DIR, transform)
val_dataset = PollutionDataset(val_df, IMG_DIR, transform)
test_dataset = PollutionDataset(test_df, IMG_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =======================
# 8. Modelo
# =======================
class PollutionModel(nn.Module):
    def __init__(self, num_classes, sensor_input_dim=3):
        super(PollutionModel, self).__init__()

        # CNN pré-treinada
        self.cnn = models.resnet18(pretrained=True)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # remover fully connected final

        # Branch sensores
        self.sensor_fc = nn.Sequential(
            nn.Linear(sensor_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        # Combinação
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

num_classes = len(label_encoder.classes_)
model = PollutionModel(num_classes)

# =======================
# 9. Treino
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, sensors, labels in train_loader:
        images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, sensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# =======================
# 10. Validação final
# =======================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, sensors, labels in test_loader:
        images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)
        outputs = model(images, sensors)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Acurácia no conjunto de teste: {100 * correct / total:.2f}%")

# =======================
# 11. Guardar modelo
# =======================
torch.save(model.state_dict(), "pollution_model.pth")
print("Modelo treinado e guardado em pollution_model.pth")
