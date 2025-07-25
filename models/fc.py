import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class MLP(nn.Module):
    """A simple multi-layer perceptron (fully connected neural network)."""

    def __init__(self, input_dim: int, output_dim: int, width: int = 50, depth: int = 3) -> None:
        super().__init__()

        input_layer = nn.Linear(input_dim, width)

        hidden_layers = []
        for i in range(depth - 2):
            hidden_layers.append(nn.Linear(width, width))
            hidden_layers.append(nn.ReLU(inplace=True))

        output_layer = nn.Linear(width, output_dim)

        self.mlp = nn.Sequential(
            input_layer,
            nn.ReLU(),
            *hidden_layers,
            output_layer,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

def fc5(input_shape: Tuple[int, ...], output_dim: int, **kwargs) -> MLP:
    input_dim = torch.prod(torch.tensor(input_shape)).item()
    return MLP(input_dim, output_dim, depth=5, **kwargs)

def fc7(input_shape: Tuple[int, ...], output_dim: int, **kwargs) -> MLP:
    input_dim = torch.prod(torch.tensor(input_shape)).item()
    return MLP(input_dim, output_dim, depth=7, **kwargs)
