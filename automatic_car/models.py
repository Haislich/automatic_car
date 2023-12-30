import torch
from torch import nn
from pathlib import Path
import sys
import numpy as np

# from time import sleep


class Model1(nn.Module):
    path = Path(sys.path[0], "models", "model1.pth")

    def __init__(self, num_channels, num_classes) -> None:
        super().__init__()
        hidden_units = 12
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=10,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=108, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        return self.linear_layer(x)

    def exists(self):
        return self.path.is_file()

    def info(self, info: str):
        path = Path(sys.path[0], "models", "model1_info")
        return path.joinpath(info)

    def predict(self, frame) -> int:
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
        frame = frame.unsqueeze(0)
        self.eval()
        prediction = 0
        with torch.inference_mode():
            softmax = nn.Softmax(dim=1)
            prediction = np.argmax(softmax(self(frame))).item()
        return prediction


class Model2(nn.Module):
    path = Path(sys.path[0], "models", "model2.pth")

    def __init__(self, num_channels, num_classes) -> None:
        super().__init__()
        hidden_units = 20
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=3),
        )
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=5),
        )
        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=7),
        )
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=20, out_features=num_classes),
        )

    def exists(self):
        return self.path.is_file()

    def info(self, info: str):
        path = Path(sys.path[0], "models", "model2_info")
        return path.joinpath(info)

    def forward(self, x: torch.Tensor):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        return self.linear_layer(x)

    def predict(self, frame) -> int:
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
        frame = frame.unsqueeze(0)
        self.eval()
        with torch.inference_mode():
            softmax = nn.Softmax(dim=1)
            prediction = np.argmax(softmax(self(frame))).item()
        return prediction
