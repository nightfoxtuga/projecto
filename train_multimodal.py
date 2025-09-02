# instruçoes uso: python3 train_multimodal.py --csv registos_labeled.csv --img_dir imagens --epochs 12 --batch_size 16


# train_multimodal.py
import os, random, argparse, json, pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ---------------- Arguments (podes ajustar) ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="registos_labeled.csv", help="CSV com image,label e colunas de sensores")
parser.add_argument("--img_dir", default="imagens", help="Pasta com imagens")
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", default="output_model", help="onde guardar modelos e artefactos")
args = parser.parse_args()

# ---------------- Seeds para reproducibilidade ----------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

os.makedirs(args.out_dir, exist_ok=True)

# ---------------- Ler CSV ----------------
df = pd.read_csv(args.csv)
# verifica colunas
required_cols = ["image", "label"]
if not all(c in df.columns for c in required_cols):
    raise SystemExit("CSV precisa de colunas 'image' e 'label'")

# identifica colunas de sensores (tudo excepto image/label)
sensor_cols = [c for c in df.columns if c not in ("image", "label")]

# remove linhas onde a imagem não existe
df["image_path"] = df["image"].apply(lambda x: os.path.join(args.img_dir, x))
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
if df.empty:
    raise SystemExit("Nenhuma imagem encontrada nas paths indicadas.")

# ---------------- Map de labels ----------------
labels = sorted(df["label"].unique())
label2idx = {l:i for i,l in enumerate(labels)}
df["label_idx"] = df["label"].map(label2idx)

# ---------------- Split estratificado ----------------
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=args.seed)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=args.seed)

print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# ---------------- Scaler para sensores ----------------
scaler = StandardScaler()
# preencher NaNs por 0 temporariamente para encaixe; idealmente calibra/limpa os teus dados
train_sensors = train_df[sensor_cols].fillna(0).values
scaler.fit(train_sensors)

# salva scaler e label map
with open(os.path.join(args.out_dir, "scaler.pkl"), "wb") as f:
    pickle.dump({"scaler":scaler, "sensor_cols": sensor_cols}, f)
with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
    json.dump({"label2idx": label2idx, "idx2label": {v:k for k,v in label2idx.items()}}, f, indent=2)

# ---------------- Transforms ----------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- Dataset ----------------
class MultimodalDataset(Dataset):
    def __init__(self, df, sensor_cols, scaler, transform=None):
        self.df = df.reset_index(drop=True)
        self.sensor_cols = sensor_cols
        self.transform = transform
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # sensor vector
        sensors = row[self.sensor_cols].fillna(0).values.astype(np.float32).reshape(1, -1)
        sensors = self.scaler.transform(sensors).reshape(-1).astype(np.float32)
        label = int(row["label_idx"])
        return img, torch.from_numpy(sensors), label

# ---------------- DataLoaders ----------------
train_ds = MultimodalDataset(train_df, sensor_cols, scaler, transform=train_transform)
val_ds   = MultimodalDataset(val_df, sensor_cols, scaler, transform=eval_transform)
test_ds  = MultimodalDataset(test_df, sensor_cols, scaler, transform=eval_transform)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ---------------- Modelo multimodal ----------------
class MultimodalNet(nn.Module):
    def __init__(self, num_sensor_feats, num_classes):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=True)
        # converte classifier para identidade para obter o vetor de características
        backbone.classifier = nn.Identity()
        self.image_model = backbone  # outputs [B, last_channel]
        self.sensor_mlp = nn.Sequential(
            nn.Linear(num_sensor_feats, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        combined = backbone.last_channel + 32
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(combined, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, sensors):
        img_feat = self.image_model(image)         # [B, C]
        sensor_feat = self.sensor_mlp(sensors)     # [B, 32]
        x = torch.cat([img_feat, sensor_feat], dim=1)
        out = self.classifier(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalNet(num_sensor_feats=len(sensor_cols), num_classes=len(labels)).to(device)

# congelar base do image_model (já feito por usar backbone.classifier=Identity; os parâmetros do backbone mantêm grad por defeito)
for param in model.image_model.parameters():
    param.requires_grad = False

# só treinar MLP sensores + classifier
params_to_optimize = list(model.sensor_mlp.parameters()) + list(model.classifier.parameters())
optimizer = optim.Adam(params_to_optimize, lr=args.lr)
criterion = nn.CrossEntropyLoss()

# ---------------- Treino com Early Stopping ----------------
best_val = 0.0
patience = 4
counter = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for images, sensors, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        images = images.to(device)
        sensors = sensors.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, sensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += images.size(0)

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total * 100.0

    # validação
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, sensors, labels in val_loader:
            images = images.to(device)
            sensors = sensors.to(device)
            labels = labels.to(device)
            outputs = model(images, sensors)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += images.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total * 100.0

    history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}: Train loss {train_loss:.4f}, Train acc {train_acc:.2f}%, Val loss {val_loss:.4f}, Val acc {val_acc:.2f}%")

    # checkpoint
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(args.out_dir, "multimodal_best.pth"))
        print("  -> Melhor modelo guardado.")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping (sem melhoria).")
            break

# ---------------- Exporta TorchScript (para Raspberry Pi) ----------------
# carregar melhor estado e traçar
model.load_state_dict(torch.load(os.path.join(args.out_dir, "multimodal_best.pth"), map_location=device))
model.eval()
# usa CPU para trace (portável para Pi)
model_cpu = model.to("cpu")
# exemplo dummy
dummy_img = torch.randn(1,3,224,224)
dummy_sensor = torch.randn(1, len(sensor_cols))
traced = torch.jit.trace(model_cpu, (dummy_img, dummy_sensor))
traced.save(os.path.join(args.out_dir, "multimodal_traced.pt"))
print("TorchScript guardado:", os.path.join(args.out_dir, "multimodal_traced.pt"))

# ---------------- Guarda history e artefactos ----------------
with open(os.path.join(args.out_dir, "train_history.pkl"), "wb") as f:
    pickle.dump(history, f)

print("Treino concluído. Artefactos em:", args.out_dir)
