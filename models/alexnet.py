import torch
import torch.nn as nn
from torch import Tensor


class AlexNet(nn.Module):
    """AlexNet model from Krizhevsky et al. (2012).
    """

    def __init__(self, input_height: int = 32, input_width: int = 32, input_channels: int = 3, channels: int = 64, num_classes: int = 100) -> None:
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, out_channels=channels, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.size = self._get_size()
        self.width = int(self.size * (1 + torch.log(torch.tensor(self.size).float()) / torch.log(torch.tensor(2.0))))

        self.classifier = nn.Sequential(
            nn.Linear(self.size, self.width),
            nn.ReLU(inplace=True),
            nn.Linear(self.width, self.width),
            nn.ReLU(inplace=True),
            nn.Linear(self.width, num_classes),
        )

    def _get_size(self) -> int:
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        return y.view(-1).size(0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(input_shape: tuple[int, int, int], output_dim: int) -> AlexNet:
    return AlexNet(input_height=input_shape[1], input_width=input_shape[2], input_channels=input_shape[0], num_classes=output_dim)
