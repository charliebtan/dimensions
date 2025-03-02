from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger

import copy

import models

from topology import fast_ripser
from utils import accuracy
from dataset import prepare_data, cycle_loader

def get_weights(net):
    return torch.cat([p.view(-1).detach().cpu() for p in net.parameters()]).numpy()

from utils import accuracy
import argparse

torch.set_float32_matmul_precision('high')

@torch.no_grad()
def evaluate(net, criterion, dataloader, batch_size=None):

    total_loss = 0.0
    total_acc = 0.0
    losses_tensors = []

    for x, y in dataloader:

        out = net(x)
        loss = criterion(out, y)

        acc = accuracy(out, y)

        total_loss += loss.sum()
        total_acc += acc

        losses_tensors.append(loss)

    mean_loss = total_loss / len(dataloader.dataset)
    mean_acc = total_acc / len(dataloader.dataset)

    return mean_loss.item(), mean_acc.item(), torch.cat(losses_tensors)

def train_step(net, opt, criterion, data):

    x, y = data

    net.train()
    opt.zero_grad()
    out = net(x)
    loss = criterion(out, y).mean()
    loss.backward()
    opt.step()

    return loss

def adversarial_pretrain(net, opt, criterion, dataloader, dataloader_eval):

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

def main():

    parser = argparse.ArgumentParser(description="Train a neural network with PH dimension analysis")

    parser.add_argument("--model", type=str, default="fc5", help="Model to use, supported: ['fc5', 'fc7', 'alexnet', 'cnn']")
    parser.add_argument("--adversarial_init", action="store_true", help="Use adversarial initialization")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use, supported: ['mnist', 'cifar10']")
    parser.add_argument("--data_path", type=str, default="~/data/", help="Path to the data")
    parser.add_argument("--max_iterations", type=int, default=10000000000, help="Maximum authorized number of iterations")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate in the experiment")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size in the experiment")
    parser.add_argument("--batch_size_eval", type=int, default=100_000, help="Batch size used for evaluation, set very high for full batch")
    parser.add_argument("--eval_freq", type=int, default=5000, help="Frequency at which we evaluate the model (training and validation sets)")
    parser.add_argument("--ripser_jump", type=int, default=20, help="Number of finite sets drawn to compute the PH dimension")
    parser.add_argument("--min_ripser_points", type=int, default=1000, help="Minimum number of points used to compute the PH dimension")
    parser.add_argument("--max_ripser_points", type=int, default=5000, help="Maximum number of points used to compute the PH dimension")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_folder", type=str, default="./results", help="Where to save the results")

    args = parser.parse_args()

    # Setup experiment
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"On device: {str(device)}")
    logger.info(f"Random seed: {args.seed}")
    torch.manual_seed(args.seed)

   # Prepare the data
    train_loader, train_loader_eval, test_loader_eval = prepare_data(args.dataset, args.data_path, args.batch_size, args.batch_size_eval)

    # Initstantiate the model
    if args.model.lower() not in ["fc5", "fc7", "alexnet", "cnn"]:
        raise NotImplementedError(f"Model {args.model} not implemented, should be in ['fc5', 'fc7', 'alexnet', 'cnn']")
    net_factory = getattr(models, args.model.lower() + "_" + args.dataset.lower())
    net = net_factory().to(device)
    logger.info("Network:")
    logger.info(net)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(reduction="none").to(device)

    # Define optimizer
    opt = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
    )

    # Lists to store the weights and losses for the persistent homology computation
    weights_list = []
    losses_list = []

    if args.adversarial_init:
        adversarial_pretrain(net, opt, criterion, train_loader, train_loader_eval)
       
    logger.info("Starting training")

    training_converged = False
    for i, (x, y) in enumerate(cycle_loader(train_loader)):

        x = x.to(device)
        y = y.to(device)

        # Take a training step
        loss = train_step(net, opt, criterion, (x, y))

        if i % 1000 == 0:
            logger.info(f"Iteration {i} - Train loss: {loss.item():.3f}")

        if torch.isnan(loss):
            logger.error('Loss has gone NaN ❌')
            return

        if i > args.max_iterations:
            training_converged = True

        # Evaluation on both datasets
        if i % args.eval_freq == 0 and not training_converged:

            # Evaluation on full training set
            train_set_loss, train_set_acc, _ = evaluate(net, criterion, train_loader_eval)

            # Evaluation on full test set
            test_set_loss, test_set_acc, _ = evaluate(net, criterion, test_loader_eval)

            logger.info(f"Evaluation at iteration {i} - Train loss: {train_set_loss:.3f}, Train acc: {train_set_acc:.3f}, Test loss: {test_set_loss:.3f}, Test acc: {test_set_acc:.3f}")

            # Stop when reach 100% training accuracy
            if (int(train_set_acc) == 100):
                logger.info(f'All training data is correctly classified in {i} iterations! ✅')
                training_converged = True

        elif training_converged:

            # Store weights for Euclidean PH dim
            weights_list.append(get_weights(net))

            # Store losses for loss-based PH dim
            train_set_loss, train_set_acc, train_losses_tensor = evaluate(net, criterion, train_loader_eval)
            logger.info(f"Computed loss tensor at iteration {i}")
            losses_list.append(train_losses_tensor)

            if (len(weights_list) == args.max_ripser_points):
                break

    # Evaluation on full test set
    test_set_loss, test_set_acc = evaluate(net, criterion, test_loader_eval)

    # Compute gaps
    loss_gap = test_set_loss - train_set_loss
    acc_gap = test_set_acc - train_set_acc

    # Compute PH dimensions
    weights_tensor = torch.tensor(weights_tensor, requires_grad=False)
    weights_np = weights_tensor.cpu().numpy()
    losses_np = torch.tensor(losses_list).cpu().numpy()

    # jump_size is a parameter of the persistent homology part
    # it defines how many finite set are drawn to perform the affine regression
    jump_size = int((args.max_ripser_points - args.min_ripser_points) / args.ripser_jump)

    logger.info("Computing euclidean PH dim...")
    ph_dim_euclidean = fast_ripser(weights_np,
                                   max_points=args.max_ripser_points,
                                   min_points=args.min_ripser_points,
                                   point_jump=jump_size)

    logger.info("Computing L1 losses based PH dim...")
    ph_dim_losses_based = fast_ripser(losses_np,
                                      max_points=args.max_ripser_points,
                                      min_points=args.min_ripser_points,
                                      point_jump=jump_size,
                                      metric="manhattan")

    # Compute weight norm
    weight_norm = np.linalg.norm(weights_np[-1]).item()

    exp_dict = {
        "ph_dim_euclidean": ph_dim_euclidean,
        "ph_dim_losses_based": ph_dim_losses_based,
        "weight_norm": weight_norm,
        "train_acc": train_set_acc,
        "eval_acc": test_set_acc,
        "acc_gap": acc_gap,
        "train_loss": train_set_loss,
        "test_loss": test_set_loss,
        "loss_gap": loss_gap,
        "learning_rate": args.lr,
        "batch_size": int(args.batch_size),
        "LB_ratio": args.lr / args.batch_size,
        "model": args.model,
        "dataset": args.dataset,
        "iterations": i,
        "seed": args.seed,
        "init": 'adversarial' if args.adversarial_init else 'random',
    }

    wandb.log(exp_dict)
    wandb.finish()


if __name__ == "__main__":
    main()