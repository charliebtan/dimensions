import torch
from loguru import logger

from utils import accuracy
from dataset import cycle_loader
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
def evaluate(net: torch.nn.Module, criterion: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float, torch.Tensor]:
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

def adversarial_pretrain(net: torch.nn.Module, opt: torch.optim.Optimizer, criterion: torch.nn.Module, dataloader: torch.utils.data.DataLoader, dataloader_eval: torch.utils.data.DataLoader) -> None:
    """Adversarial pretraining of the network on fixed but shuffled labels.
    See https://arxiv.org/abs/1906.02613 for more details."""

    logger.info("Starting adversarial pretraining")

    # Randomize the labels at the start of training
    random_indices = torch.randperm(len(dataloader_eval.dataset))
    dataloader.dataset.y = dataloader.dataset.y[random_indices]
    dataloader_eval.dataset.y = dataloader_eval.dataset.y[random_indices]

    for i, (x, y) in enumerate(cycle_loader(dataloader)):

        train_step(net, opt, criterion, (x, y))

        if i % 1000 == 0:

            # We don't use the loss from training, but re-compute it on the full set with fixed parameters

            loss, acc, _ = evaluate(net, criterion, dataloader_eval)
            logger.info(f"Adversarial pretraining iteration {i} - Loss: {loss}, Acc: {acc}")

            if int(acc) == 100:
                logger.info(f'All random training data is correctly classified in {i} iterations! ✅')
                break 

    # Restore the original labels
    inverse_random_indices = torch.argsort(random_indices)
    dataloader.dataset.y = dataloader.dataset.y[inverse_random_indices]
    dataloader_eval.dataset.y = dataloader_eval.dataset.y[inverse_random_indices]
