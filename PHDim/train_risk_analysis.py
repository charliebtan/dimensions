from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger

from models import alexnet, fc_mnist, vgg, fc_cifar, lenet
from models.vgg import vgg as make_vgg
from topology import fast_ripser
from utils import accuracy
from PHDim.dataset import get_data_simple
from PHDim.eval import eval, recover_eval_tensors, eval_on_tensors

from PHDim.hausdorff_alpha import estimator_vector_full, estimator_vector_projected


class UnknownWeightFormatError(BaseException):
    ...


def get_weights(net):
    with torch.no_grad():
        w = []

        # TODO: improve this?
        for p in net.parameters():
            w.append(p.view(-1).detach().to(torch.device('cpu')))
        return torch.cat(w).cpu().numpy()


def main(iterations: int = 10000000,
         batch_size_train: int = 100,
         batch_size_eval: int = 1000,
         lr: float = 1.e-1,
         eval_freq: int = 1000,
         dataset: str = "mnist",
         data_path: str = "~/data/",
         model: str = "fc",
         depth: int = 5,
         width: int = 50,
         optim: str = "SGD",
         min_points: int = 200,
         seed: int = 42,
         save_weights_file: str = None,
         compute_dimensions: bool = True,
         ripser_points: int = 1000,
         jump: int = 20,
         ):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info(f"on device {str(device)}")
    logger.info(f"Random seed ('torch.manual_seed'): {seed}")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    # training setup
    if dataset not in ["mnist", "cifar10"]:
        raise NotImplementedError(f"Dataset {dataset} not implemented, should be in ['mnist', 'cifar10']")
    train_loader, test_loader_eval, train_loader_eval, num_classes = get_data_simple(dataset,
                                                         data_path,
                                                         batch_size_train,
                                                         batch_size_eval,
                                                         subset=1.0)

    # TODO: use the args here
    if model not in ["fc", "alexnet", "vgg", "lenet"]:
        raise NotImplementedError(f"Model {model} not implemented, should be in ['fc', 'alexnet', 'vgg']")
    if model == 'fc':
        if dataset == 'mnist':
            input_size = 28**2
            net = fc_mnist(input_dim=input_size, width=width, depth=depth, num_classes=num_classes).to(device)
        elif dataset == 'cifar10':
            net = cnn_cifar().to(device)
    elif model == 'alexnet':
        if dataset == 'mnist':
            net = alexnet(input_height=28, input_width=28, input_channels=1, num_classes=num_classes).to(device)
        else:
            net = alexnet(ch=64, num_classes=num_classes).to(device)
    elif model == 'vgg':
        net = make_vgg(depth=depth, num_classes=num_classes, batch_norm=False).to(device)
    elif model == "lenet":
        if dataset == "mnist":
            net = lenet(input_channels=1, height=28, width=28).to(device)
        else:
            net = lenet().to(device)

    logger.info("Network: ")
    print(net)

    net = net.to(device)

    crit = nn.CrossEntropyLoss().to(device)
    crit_unreduced = nn.CrossEntropyLoss(reduction="none").to(device)

    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)

    # Recovering evaluation tensors (made to speed up the experiment)
    eval_x, eval_y = recover_eval_tensors(train_loader_eval)
    test_x, test_y = recover_eval_tensors(test_loader_eval)

    # initialize results of the experiment, returned if didn't work
    exp_dict = {}

    # weights
    weights_history = deque([])
    loss_history = deque([])

    STOP = False  # Do we have enough point for persistent homology
    CONVERGED = False  # has the experiment converged?

    # Defining optimizer
    opt = getattr(torch.optim, optim)(
        net.parameters(),
        lr=lr,
    )

    logger.info("Starting training")
    for i, (x, y) in enumerate(circ_train_loader):

        # Sequentially running evaluation step
        # first record is at the initial point
        if i % eval_freq == 0 and (not CONVERGED):

            with torch.no_grad():

                # Evaluation on validation set
                logger.info(f"Evaluation at iteration {i}")
                te_hist, *_ = eval(test_loader_eval, net, crit_unreduced, opt)
                logger.info(f"Evaluation on test set at iteration {i} finished ✅, accuracy: {round(te_hist[1], 3)}")

                # Evaluation on training set
                tr_hist, losses, _ = eval(train_loader_eval, net, crit_unreduced, opt)
                logger.info(f"Training accuracy at iteration {i}: {round(tr_hist[1], 3)}%")

                # Stopping criterion based on 100% accuracy
                if (int(tr_hist[1]) == 100) and (CONVERGED is False):
                    logger.info(f'All training data is correctly classified in {i} iterations! ✅')
                    CONVERGED = True

                logger.info(f"Loss sum at iteration{i}: {losses.sum().item()}")

        net.train()

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)

        if i % 1000 == 0:
            logger.info(f"Loss at iteration {i}: {loss.item()}")

        if torch.isnan(loss):
            logger.error('Loss has gone nan ❌')
            break

        if tr_hist[1] < 20 and i > 100000:
            logger.error('Training accuracy is below 20% - not converging ❌')
            break

        # calculate the gradients
        loss.backward()

        # take the step
        opt.step()

        if i > iterations:
            CONVERGED = True
            if not compute_dimensions:
                STOP=True

        if CONVERGED:
            weights_history.append(get_weights(net))

            if compute_dimensions:
                tr_hist, losses, _ = eval_on_tensors(eval_x, eval_y, net, crit_unreduced)
                loss_history.append(losses.cpu())

            # Validation history
            te_hist, _, _ = eval_on_tensors(test_x, test_y, net, crit_unreduced)

        # clear cache
        torch.cuda.empty_cache()

        if (len(weights_history) > ripser_points) and compute_dimensions:
            STOP = True
            weights_history.popleft()
            loss_history.popleft()

        # final evaluation and saving results
        if STOP and CONVERGED:

            # if no convergence, we don't record
            if len(weights_history) < ripser_points - 1:
                logger.warning("Experiment did not converge")
                break

            with torch.no_grad():

                # Some logging
                logger.debug('eval time {}'.format(i))
                te_hist, *_ = eval(test_loader_eval, net, crit_unreduced, opt)
                tr_hist, *_ = eval(train_loader_eval, net, crit_unreduced, opt)

                # Turn collected iterates (both weights and losses) into numpy arrays
                if compute_dimensions:
                    weights_history_np = np.stack(tuple(weights_history))
                    del weights_history

                loss_history_np = torch.stack(tuple(loss_history)).cpu().numpy()

                # jump_size is a parameter of the persistent homology part
                # Ijump defines how many finite set are drawn to perform the affine regression
                jump_size = int((ripser_points - min_points) / jump)

                if compute_dimensions:

                    logger.info("Computing euclidean PH dim...")
                    ph_dim_euclidean = fast_ripser(weights_history_np,
                                                   max_points=ripser_points,
                                                   min_points=min_points,
                                                   point_jump=jump_size)

                    logger.info("Computing L1 losses based PH dim...")
                    ph_dim_losses_based = fast_ripser(loss_history_np,
                                                      max_points=ripser_points,
                                                      min_points=min_points,
                                                      point_jump=jump_size,
                                                      metric="manhattan")

                    traj = torch.tensor(weights_history_np, requires_grad=False)

                    alpha_full_5000 = estimator_vector_full(traj)
                    alpha_proj_med_5000, alpha_proj_max_5000 = estimator_vector_projected(traj)

                    traj_epoch = traj[-len(train_loader):]

                    alpha_full_epoch = estimator_vector_full(traj_epoch)
                    alpha_proj_med_epoch, alpha_proj_max_epoch = estimator_vector_projected(traj_epoch)

                    # the std deviation of all points from the centroid
                    std_dist = torch.sqrt(torch.sum(torch.var(torch.tensor(traj), dim=0))).item()
                    norm = np.linalg.norm(traj[-1]).item()

                    step_sizes = [] # need to start with None as no step size for first point

                    for q in range(1, traj.shape[0]):

                        gradient_update = traj[q] - traj[q-1] # difference between points
                        step_sizes.append(torch.norm(gradient_update)) # euclidean distance between points

                    mean_step_size = np.mean(step_sizes)

                else:
                    ph_dim_euclidean = None
                    ph_dim_losses_based = None
                    alpha_full_5000 = None
                    alpha_proj_med_5000 = None
                    alpha_proj_max_5000 = None
                    alpha_full_epoch = None
                    alpha_proj_med_epoch = None
                    alpha_proj_max_epoch = None
                    std_dist = None
                    norm = None
                    mean_step_size = None

            exp_dict = {
                "ph_dim_euclidean": ph_dim_euclidean,
                "ph_dim_losses_based": ph_dim_losses_based,
                "alpha_full_5000" : alpha_full_5000,
                "alpha_proj_med_5000": alpha_proj_med_5000,
                "alpha_proj_max_5000": alpha_proj_max_5000,
                "alpha_full_epoch": alpha_full_epoch,
                "alpha_proj_med_epoch": alpha_proj_med_epoch,
                "alpha_proj_max_epoch": alpha_proj_max_epoch,
                "std_dist": std_dist,
                "norm": norm,
                "step_size": mean_step_size,
                "train_acc": tr_hist[1],
                "eval_acc": te_hist[1],
                "acc_gap": tr_hist[1] - te_hist[1],
                "train_loss": tr_hist[0],
                "test_loss": te_hist[0],
                "loss_gap": te_hist[0] - tr_hist[0],
                "learning_rate": lr,
                "batch_size": int(batch_size_train),
                "LB_ratio": lr / batch_size_train,
                "depth": depth,
                "width": width,
                "model": model,
                "iterations": i,
                "seed": seed,
                "dataset": dataset,
            }
            break

        # Saving weights if specified
        if save_weights_file is not None:
            # logger.info(f"Saving last weights in {str(save_weights_file)}")
            torch.save(net.state_dict(), str(save_weights_file))
            exp_dict["saved_weights"] = str(save_weights_file)

    return exp_dict
