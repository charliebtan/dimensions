import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, width=50, depth=3):
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

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

def fc5(input_shape, output_dim):
    input_dim = torch.prod(torch.tensor(input_shape))
    return MLP(input_dim, output_dim, depth=5)

def fc7(input_shape, output_dim):
    input_dim = torch.prod(torch.tensor(input_shape))
    return MLP(input_dim, output_dim, depth=7)
