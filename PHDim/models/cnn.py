import torch
from torch import Tensor
from typing import Tuple

import torch.nn as nn

class CNN(nn.Module):
    """A simple convolutional neural network defined by 
    https://arxiv.org/pdf/1912.02292 as "standard CNN."
    """

    def __init__(self, base_width: int = 64, input_shape: Tuple[int, int, int] = (3, 32, 32), num_classes: int = 10) -> None:
        super().__init__()

        widths = [input_shape[0]] + [base_width * 2**i for i in range(4)]
        max_pools = [1, 2, 2, 8]

        layers = []
        for i in range(4):
            layers.append(nn.Conv2d(widths[i], widths[i + 1], 3, padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(max_pools[i]))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(widths[-1], num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return self.linear(x.flatten(start_dim=1))

def cnn(input_shape: Tuple[int, int, int], num_classes: int) -> CNN:
    return CNN(input_shape=input_shape, num_classes=num_classes)
