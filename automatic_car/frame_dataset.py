import os
import sys
from pathlib import Path
from torch import Tensor
from torchvision.io import read_image
from torch.utils.data import Dataset


class FramesDataset(Dataset):
    def __init__(self, dataset_type: str) -> None:
        path = Path(sys.path[0], "dataset", dataset_type)
        if not path.is_dir():
            raise FileNotFoundError(f"Unable to find the {dataset_type} dataset.")
        self.labels = []
        self.images = []
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
        return image, label


# FramesDataset("train")
