import torch.nn as nn
from fcn import FCN


class CNN(nn.Module):
    def __init__(self, input_shape, channels, max_pools, padding, lin_in_dim, num_classes, use_bias=True, use_bn=False):
        super().__init__()

        assert len(channels) == len(max_pools)
        assert len(channels) == len(padding)
        
        channels = [input_shape[0]] + channels

        conv_bias = use_bias and not use_bn

        layers = []
        for i in range(len(max_pools)):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], 3, padding=padding[i], bias=conv_bias))
            if use_bn:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU())
            if max_pools[i] > 1:
                layers.append(nn.MaxPool2d(max_pools[i]))
        self.layers = nn.ModuleList(layers)

        self.linear = nn.Linear(lin_in_dim, 10, bias=use_bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


def cnn_mnist(num_classes, use_bias, use_bn):
    input_shape = (1, 28, 28)
    widths = [64, 64]
    max_pools = [2, 2]
    padding = ['same'] * 2
    net = CNN(input_shape, widths, max_pools, padding, 3136, 10, use_bias=use_bias, use_bn=use_bn)
    return net


def cnn_cifar(num_classes, use_bias, use_bn):
    input_shape = (3, 32, 32)
    widths = [64, 64, 64, 64, 64, 512]
    max_pools = [1, 2, 1, 2, 1, 8]
    padding = ['same'] * 6
    net = CNN(input_shape, widths, max_pools, padding, 512, 10, use_bias=use_bias, use_bn=use_bn)
    return net
