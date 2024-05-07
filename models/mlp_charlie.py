import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_shape, hidden_dims, num_classes, use_bias):
        super().__init__()

        input_dim = torch.prod(torch.tensor(input_shape))

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0], bias=use_bias))
        layers.append(nn.ReLU())
        for i in range(0, len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=use_bias))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], num_classes, bias=use_bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        return x


def mlp_mnist(use_bias=True):
    input_shape = (1, 28, 28)
    hidden_dims = [128, 128]
    num_classes = 10
    return MLP(input_shape, hidden_dims, num_classes, use_bias)

def mlp_chd(use_bias=True):
    input_shape = (10)
    hidden_dims = [128, 128]
    num_classes = 1
    return MLP(input_shape, hidden_dims, num_classes, use_bias)
