import torch

# import torchvision
# from torchvision import transforms
from torch import nn

# Check on which device computations needs to be done
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SHAPE = (96, 96, 3)
OUTPUT_SHAPE = 5


# input size = (96, 96, 3)
class BaseLine(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # self.cnn_layers = nn.Sequential(nn.Conv3d())
