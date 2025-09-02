# evaluate.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from dataset_and_model import (
    PollutionDataset, PollutionModel,
    label_encoder, test_df, IMG_DIR, transform
)

# =======================
# 1. Preparar dataset de teste
# =======================
test_dataset = PollutionDataset(test_df, IMG_DIR, transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# =======================
# 2. Carregar modelo
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label_encoder.classes_)

model = PollutionModel(num_classes)
model.load_state_dict(torch.load("pollution_model.pth", map_location=device))
model.to(device)
model.eval()

# =======================
# 3. Avaliação
# =======================
all_labels = []
all_preds = []
all_probs = []  # guardar probabilidades

with torch.no_grad():
    for images, sensors, labels in test_loader:
        images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)
        outputs = model(images, sensors)

        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# =======================
# 4. Relatório e Confusion Matrix
# =======================
print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

cm = confusion_matrix(all_labels, all_preds)
print("\n=== Confusion Matrix ===")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# =======================
# 5. Análise de Erros
# =======================
errors_idx = [i for i, (t, p) in enumerate(zip(all_labels, all_preds)) if t != p]

print(f"\nTotal de erros: {len(errors_idx)}")

# criar pasta para salvar erros
error_dir = "errors"
if os.path.exists(error_dir):
    shutil.rmtree(error_dir)
os.makedirs(error_dir, exist_ok=True)

for idx in errors_idx:
    row = test_df.iloc[idx]
    true_label = label_encoder.classes_[row["qa"]]
    pred_label = label_encoder.classes_[all_preds[idx]]

    img_name = str(row["image_name"]) + ".jpg"
    src = os.path.join(IMG_DIR, img_name)
    dst = os.path.join(error_dir, f"true_{true_label}_pred_{pred_label}_{img_name}")
    if os.path.exists(src):
        shutil.copy(src, dst)

print(f"Imagens de erros copiadas para pasta: {error_dir}")

# =======================
# 6. Curvas ROC/AUC (One-vs-All)
# =======================
from sklearn.preprocessing import label_binarize

y_true_bin = label_binarize(all_labels, classes=range(num_classes))
y_score = np.array(all_probs)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC
plt.figure(figsize=(7, 6))
for i, class_name in enumerate(label_encoder.classes_):
    plt.plot(fpr[i], tpr[i], label=f"{class_name} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - One vs All")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
