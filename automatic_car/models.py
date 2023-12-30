import torch
from torch import nn
from pathlib import Path
import sys


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
