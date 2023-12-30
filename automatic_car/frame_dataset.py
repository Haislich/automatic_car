import os
import sys
from pathlib import Path
from torch import Tensor
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import numpy as np
from collections import defaultdict


class FramesDataset(Dataset):
    __num_channels = 3
    __num_classes = 5
    label_map = {
        0: "Do nothing",
        1: "Steer left",
        2: "Steer right",
        3: "Gas",
        4: "Brake",
    }

    def __init__(self, dataset_type: str) -> None:
        path = Path(sys.path[0], "data", dataset_type)
        if not path.is_dir():
            raise FileNotFoundError(f"Unable to find the {dataset_type} dataset.")
        self.labels: list[int] = []
        self.images: list[str] = []
        for c in os.listdir(path):
            file_path = path.joinpath(c)
            for image_name in os.listdir(file_path):
                image_path = file_path.joinpath(image_name)
                self.images.append(str(image_path))
                self.labels.append(int(c))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> tuple[Tensor, int]:
        image = read_image(self.images[index])
        label = self.labels[index]
        return image.to(torch.float), label

    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch
    def get_weights(self):
        tmp = defaultdict(list)
        for label, image in zip(self.labels, self.images):
            tmp[label].append(image)
        weights = np.array([len(tmp[label]) for label in tmp], dtype=np.float32)
        weights = np.max(weights) / weights
        return torch.tensor(weights, dtype=torch.float32)

    def get_examples_per_class(self):
        tmp: dict[int, list[str]] = defaultdict(list[str])
        for label, image in zip(self.labels, self.images):
            tmp[label].append(image)
        return {k: len(v) for k, v in tmp.items()}

    def get_num_channels(self):
        return self.__num_channels

    def get_num_classes(self):
        return self.__num_classes


class ModifiedFramesDataset(FramesDataset):
    def __init__(self, dataset_type: str) -> None:
        super().__init__(dataset_type)

    def __getitem__(self, index) -> tuple[Tensor, int]:
        image = read_image(self.images[index])
        label = self.labels[index]
        return image[:, 84:96, :], label
