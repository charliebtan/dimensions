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

def fc5_mnist():
    input_dim = 28 * 28
    output_dim = 10
    return MLP(input_dim, output_dim, depth=5)

def fc7_mnist(): 
    input_dim = 28 * 28
    output_dim = 10
    return MLP(input_dim, output_dim, depth=7)

def fc5_california():
    input_dim = 13
    output_dim = 1
    return MLP(input_dim, output_dim, depth=5)