import datetime
import json
import os
from collections import deque
from pathlib import Path

import fire
import numpy as np
import torch
import wandb
from loguru import logger
from pydantic import BaseModel


from models import fc_bhp

import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger
import argparse

import models
from topology import fast_ripser
from train_utils import train_step, evaluate_regressor
from utils import get_weights
from data import prepare_data_regression, loop_dataloader


torch.set_float32_matmul_precision('high')

def build_model(model_name, dataset_name):

    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    # Initstantiate the model
    if model_name not in ["fc5", "fc7"]:
        raise NotImplementedError(f"Model {model_name} not implemented, should be in ['fc5', 'fc7']")

    if dataset_name == "california":
        input_shape = (1, 28, 28)
        output_dim = 10
        width = 200
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented, should be in ['california']")

    net_factory = getattr(models, model_name)
    net = net_factory(input_shape, output_dim, width)

    logger.info("Network:")
    logger.info(net)

    return net

def main():

    parser = argparse.ArgumentParser(description="Train a neural network with PH dimension analysis")

    parser.add_argument("--model", type=str, default="fc5", help="Model to use, supported: ['fc5', 'fc7']")
    parser.add_argument("--dataset", type=str, default="california", help="Dataset to use, supported: ['california']")
    parser.add_argument("--data_path", type=str, default="~/data/", help="Path to the data")
    parser.add_argument("--max_iterations", type=int, default=1e7, help="Maximum authorized number of iterations")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate in the experiment")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size in the experiment")
    parser.add_argument("--batch_size_eval", type=int, default=100_000, help="Batch size used for evaluation, set very high for full batch")
    parser.add_argument("--eval_freq", type=int, default=2000, help="Frequency at which we evaluate the model (training and validation sets)")
    parser.add_argument("--ripser_jump", type=int, default=20, help="Number of finite sets drawn to compute the PH dimension")
    parser.add_argument("--min_ripser_points", type=int, default=1000, help="Minimum number of points used to compute the PH dimension")
    parser.add_argument("--max_ripser_points", type=int, default=5000, help="Maximum number of points used to compute the PH dimension")
    parser.add_argument("--stopping_criterion", type=float, default=5e-3, help="Stopping criterion for convergence")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_folder", type=str, default="./results", help="Where to save the results")

    args = parser.parse_args()

    wandb.init()
    wandb.config.update(args)

    # Setup experiment
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"On device: {str(device)}")
    logger.info(f"Random seed: {args.seed}")
    torch.manual_seed(args.seed)

    # Prepare the data
    train_loader, train_dataset, test_dataset = prepare_data_regression(args.dataset, args.data_path, args.batch_size)

    # Build the model
    net = build_model(args.model, args.dataset).to(device)

    # Define the loss function
    criterion = nn.MSELoss(reduction='none').to(device)

    # Define optimizer
    opt = torch.optim.SGD(net.parameters(), lr=args.lr)

    # Lists to store the weights and losses for the persistent homology computation
    weights_list = []
    losses_list = []

    logger.info("Starting training")

    previous_train_set_loss = 1e9
    training_converged = False
    for i, (x, y) in enumerate(loop_dataloader(train_loader)):

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
            train_set_loss, _ = evaluate_regressor(net, criterion, train_dataset)

            # Evaluation on full test set
            test_set_loss, _ = evaluate_regressor(net, criterion, test_dataset)
            logger.info(f"Evaluation at iteration {i} - Train loss: {train_set_loss:.3f}, Test loss: {test_set_loss:.3f}")

            # Stop when reach convergence criterion
            if (i > 0) and (np.abs(train_set_loss - previous_train_set_loss) / previous_train_loss < args.stopping_criterion):
                logger.info(f"Experiment converged in {i} iterations")
                training_converged = True

            previous_train_loss = train_set_loss

        elif training_converged:

            # Store weights for Euclidean PH dim
            weights_list.append(get_weights(net))

            # Store losses for loss-based PH dim
            _, train_losses_tensor = evaluate_regressor(net, criterion, train_dataset)
            logger.info(f"Computed loss tensor at iteration {i}")
            losses_list.append(train_losses_tensor)

            if len(weights_list) == args.max_ripser_points:
                break

    # Evaluation on full train + test set
    train_set_loss, _ = evaluate_regressor(net, criterion, train_dataset)
    test_set_loss, _ = evaluate_regressor(net, criterion, test_dataset)

    # Compute gap
    loss_gap = test_set_loss - train_set_loss

    # Compute PH dimensions
    weights_tensor = torch.tensor(weights_tensor, requires_grad=False)
    weights_np = weights_tensor.cpu().numpy()
    losses_np = torch.tensor(losses_list).cpu().numpy()

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
        "train_loss": train_set_loss,
        "test_loss": test_set_loss,
        "loss_gap": loss_gap,
        "iterations": i,
    }

    wandb.log(exp_dict)
    wandb.finish()

if __name__ == "__main__":
    main()
