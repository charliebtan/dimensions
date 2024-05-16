import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, base_width=64, input_shape=(3, 32, 32), num_classes=10, use_bn=False):
        super().__init__()

        widths = [input_shape[0]] + [base_width * 2**i for i in range(4)]
        max_pools = [1, 2, 2, 8]

        layers = []
        for i in range(4):
            layers.append(nn.Conv2d(widths[i], widths[i + 1], 3, padding="same"))
            if use_bn:
                layers.append(nn.BatchNorm2d(widths[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(max_pools[i]))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.linear(x.flatten(start_dim=1))

def cnn():
    return CNN()