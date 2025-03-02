import time

import torch
import torch.nn as nn
from loguru import logger

from utils import accuracy


@torch.no_grad()
def get_dataloader_as_tensors(dataloader):

    all_x, all_y = [], []

    for x, y in dataloader:

        all_x.append(x)
        all_y.append(y)

    return torch.cat(all_x), torch.cat(all_y)


@torch.no_grad()
def evaluate(x, y, net, criterion, batch_size=None):

    if batch_size is None:
        batch_size = x.shape[0]

    for i in range(0, x.shape[0], batch_size):

        x_batch = x[i : i + batch_size]
        y_batch = y[i : i + batch_size]

        out = net(x_batch)
        loss = criterion(out, y_batch)
        acc = accuracy(out, y_batch)

        if i == 0:
            total_loss = loss
            total_acc = acc
        else:
            total_loss += loss * x_batch.shape[0]
            total_acc += acc * x_batch.shape[0]

    return total_loss / x.shape[0], total_acc / x.shape[0]
