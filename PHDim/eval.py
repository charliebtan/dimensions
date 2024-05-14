import time

import torch
import torch.nn as nn
from loguru import logger

from utils import accuracy


@torch.no_grad()
def recover_eval_tensors(dataloader):

    final_x, final_y = [], []

    for x, y in dataloader:

        final_x.append(x)
        final_y.append(y)

    return torch.cat(final_x, 0), torch.cat(final_y, 0)


@torch.no_grad()
def eval_on_tensors(x, y, net, criterion):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    x, y = x.to(device), y.to(device)

    out = net(x)

    losses = criterion(out, y)
    prec = accuracy(out, y)

    hist = [
        losses.sum().item() / x.shape[0],
        prec,
    ]

    return hist, losses, out


@torch.no_grad()
def eval(eval_loader, net, criterion, opt, eval: bool = False):
    """
    WARNING: criterion is not used anymore
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net.eval()

    # run over both test and train set
    total_size = torch.tensor([0], device=device)
    total_loss = torch.tensor([0.0], device=device)
    total_acc = torch.tensor([0.0], device=device)
    losses = []
    outputs = []

    for x, y in eval_loader:

        # loop over dataset
        x, y = x.to(device), y.to(device)

        out = net(x)

        losses_unreduced = criterion(out, y)
        prec = accuracy(out, y)
        bs = x.size(0)

        total_size += bs
        total_loss += losses_unreduced.sum()
        total_acc += prec * bs

        losses.append(losses_unreduced)
        outputs.append(out.flatten())

    hist = [
        (total_loss / total_size).item(),
        (total_acc / total_size).item(),
    ]

    # losses: list of tensors of shape (batch_size)
    # We concatenate it into a tensor of shape (len(eval_loader))
    losses = torch.cat(losses)
    outputs = torch.cat(outputs)

    return hist, losses, outputs
