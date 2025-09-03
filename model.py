# model.py
import torch
import torch.nn as nn
from torchvision import models

class PollutionModel(nn.Module):
    def __init__(self, num_classes, sensor_input_dim=3):
        super(PollutionModel, self).__init__()

        # CNN pré-treinada
        self.cnn = models.resnet18(pretrained=False)  # sem baixar durante deploy
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

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
