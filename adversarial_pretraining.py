import torch
from loguru import logger

from training_utils import train_step, evaluate_classifier

from utils import accuracy
from data import loop_dataloader
from typing import Tuple, List

def adversarial_pretrain(net: torch.nn.Module, opt: torch.optim.Optimizer, criterion: torch.nn.Module, dataloader: torch.utils.data.DataLoader, dataloader_eval: torch.utils.data.DataLoader) -> None:
    """Adversarial pretraining of the network on fixed but shuffled labels.
    See https://arxiv.org/abs/1906.02613 for more details."""

    logger.info("Starting adversarial pretraining")

    # Randomize the labels at the start of training
    random_indices = torch.randperm(len(dataloader_eval.dataset))
    dataloader.dataset.y = dataloader.dataset.y[random_indices]
    dataloader_eval.dataset.y = dataloader_eval.dataset.y[random_indices]

    for i, (x, y) in enumerate(loop_dataloader(dataloader)):

        train_step(net, opt, criterion, (x, y))

        if i % 1000 == 0:

            # We don't use the loss from training, but re-compute it on the full set with fixed parameters

            loss, acc, _ = evaluate_classifier(net, criterion, dataloader_eval)
            logger.info(f"Adversarial pretraining iteration {i} - Loss: {loss}, Acc: {acc}")

            if int(acc) == 100:
                logger.info(f'All random training data is correctly classified in {i} iterations! ✅')
                break 

    # Restore the original labels
    inverse_random_indices = torch.argsort(random_indices)
    dataloader.dataset.y = dataloader.dataset.y[inverse_random_indices]
    dataloader_eval.dataset.y = dataloader_eval.dataset.y[inverse_random_indices]
