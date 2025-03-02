import torch
from loguru import logger

from utils import accuracy
from data import loop_dataloader
from typing import Tuple, List

def train_step(net: torch.nn.Module, opt: torch.optim.Optimizer, criterion: torch.nn.Module, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """One step of training."""

    x, y = data

    net.train()
    opt.zero_grad()
    out = net(x)
    loss = criterion(out, y).mean()
    loss.backward()
    opt.step()

    return loss

@torch.no_grad()
def evaluate_classifier(net: torch.nn.Module, criterion: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float, torch.Tensor]:
    """Evaluate the network on a dataset, returning the mean loss, 
    mean accuracy and tensor of all losses."""

    total_loss = 0.0
    total_acc = 0.0
    losses_tensors: List[torch.Tensor] = []

    for x, y in dataloader:

        out = net(x)
        loss = criterion(out, y)

        acc = accuracy(out, y)

        total_loss += loss.sum()
        total_acc += acc

        losses_tensors.append(loss)

    mean_loss = (total_loss / len(dataloader.dataset)).item()
    mean_acc = total_acc / len(dataloader.dataset)

    return mean_loss, mean_acc, torch.cat(losses_tensors)

@torch.no_grad()
def evalulate_regressor(net: torch.nn.Module, criterion: torch.nn.Module, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, torch.Tensor]:
    """Evaluate the regressor on a dataset, returning the mean loss and tensor of all losses."""

    net.eval()

    x, y = data
    pred = net(x)
    loss = losses.mean().cpu().item()
    losses = criterion(pred, y)

    return loss, losses

