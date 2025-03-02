from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger

from models import alexnet, fc_mnist, vgg, fc_cifar, lenet, cnn
from models.vgg import vgg as make_vgg
from topology import fast_ripser
from utils import accuracy
from PHDim.dataset import get_data_simple
from PHDim.eval import eval, recover_eval_tensors, eval_on_tensors

from PHDim.hausdorff_alpha import estimator_vector_full, estimator_vector_projected

def get_weights(net):
    return torch.cat([p.view(-1).detach().cpu() for p in net.parameters()]).numpy()

from utils import accuracy


@torch.no_grad()
def get_dataloader_as_tensors(dataloader):

    all_x, all_y = [], []

    for x, y in dataloader:

        all_x.append(x)
        all_y.append(y)

    return torch.cat(all_x), torch.cat(all_y)


@torch.no_grad()
def evaluate(net, criterion, x, y, batch_size=None):

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

def train_step(net, x, y, opt):

    net.train()
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    opt.zero_grad()
    out = net(x)
    loss = crit(out, y)
    loss.backward()
    opt.step()

    return loss

def adversarial_pretrain(net, opt, criterion, train_loader):

    logger.info("Starting adversarial pretraining")

    train_loader_random = None
    cycle_train_loader_random = cycle_loader(train_loader_random)
    x_random, y_random = get_dataloader_as_tensors(train_loader_random)

    x_random, y_random = x_random.to(device), y_random.to(device)

    for i, (x, y) in enumerate(cycle_train_loader_random):

        train_step(net, x, y, opt)

        if i % 1000 == 0:

            # we don't use the loss from training, but re-compute it for fixed parameters

            loss, acc = evaluate(net, criterion, x_random, y_random)
            logger.info(f"Iteration: {i}, Loss: {loss}, Acc: {acc}")

            if int(acc) == 100:
                logger.ingo(f'All random training data is correctly classified in {i} iterations! ✅')
                break 

def train():

    for i, (x, y) in enumerate(circ_train_loader):

        if i % eval_freq == 0 and (not CONVERGED):

            logger.info(f"Evaluation at iteration {i}")

            # Evaluation on full training set
            train_loss, train_acc = evaluate(net, criterion, train_x, train_y)
            logger.info(f"Training accuracy at iteration {i}: {round(train_acc, 3)}%")

            # Evaluation on full test set
            test_loss, test_acc = evaluate(net, criterion, test_x, test_y)
            logger.info(f"Evaluation on test set at iteration {i} finished ✅, accuracy: {round(test_acc, 3)}")

            # Stop when reach 100% training accuracy
            if (int(train_acc) == 100):
                logger.info(f'All training data is correctly classified in {i} iterations! ✅')
                CONVERGED = True

            logger.info(f"Loss sum at iteration{i}: {losses.sum().item()}")

        loss = train_step(net, x, y, opt)

        if i % 1000 == 0:
            logger.info(f"Loss at iteration {i}: {loss.item()}")

        if torch.isnan(loss):
            logger.error('Loss has gone nan ❌')
            break

        if i > iterations:
            CONVERGED = True

        if CONVERGED:

            weights_tensor.append(get_weights(net))

            # TODO do I need all of this??

            if not model == 'cnn':
                tr_hist, losses, _ = eval_on_tensors(eval_x, eval_y, net, crit_unreduced)
            else:
                tr_hist, losses, _ = eval(train_loader_eval, net, crit_unreduced, opt)
            loss_history.append(losses.cpu())

            # Validation history
            if not model == 'cnn':
                te_hist, _, _ = eval_on_tensors(test_x, test_y, net, crit_unreduced)
            else:
                te_hist, _, _ = eval(test_loader_eval, net, crit_unreduced, opt)

        else:
            if tr_hist[1] < 15 and i > 1000000:
                logger.error('Training accuracy is below 20% - not converging ❌')
                break

        # clear cache
        torch.cuda.empty_cache()

 

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
         ripser_points: int = 1000,
         jump: int = 20,
         random: bool = False,
         ):

   # training setup
    if dataset not in ["mnist", "cifar10"]:
        raise NotImplementedError(f"Dataset {dataset} not implemented, should be in ['mnist', 'cifar10']")
    train_loader, test_loader_eval, train_loader_eval, num_classes, train_loader_random, train_loader_random_eval = get_data_simple(dataset,
                                                         data_path,
                                                         batch_size_train,
                                                         batch_size_eval,
                                                         subset=1.0)

    # TODO: use the args here
    if model not in ["fc", "alexnet", "vgg", "lenet", 'cnn']:
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
    elif model == 'cnn':
        net = cnn().to(device)
    elif model == 'vgg':
        net = make_vgg(depth=depth, num_classes=num_classes, batch_norm=False).to(device)
    elif model == "lenet":
        if dataset == "mnist":
            net = lenet(input_channels=1, height=28, width=28).to(device)
        else:
            net = lenet().to(device)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info(f"On device: {str(device)}")
    logger.info(f"Random seed: {seed}")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    logger.info("Network:")
    print(net)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

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
    weights_tensor = deque([])
    loss_history = deque([])

    STOP = False  # Do we have enough point for persistent homology
    CONVERGED = False  # has the experiment converged?
    DIM_NOW = False
    END = False

    if not random:
        ph_dim_euclidean = []
        ph_dim_losses_based = []
        std_dist = []
        norm = []
        mean_step_size = []
        tr_acc = []
        te_acc = []
        acc_gap = []
        tr_loss = []
        te_loss = []
        loss_gap = []

    # Defining optimizer
    opt = getattr(torch.optim, optim)(
        net.parameters(),
        lr=lr,
    )


 
    if adversarial_init:
        adversarial_pretrain()
       
    logger.info("Starting training")

    train()



        if (len(weights_tensor) == ripser_points):

            # Evaluation on full training set
            train_loss, train_acc = evaluate(net, criterion, train_x, train_y)

            # Evaluation on full test set
            test_loss, test_acc = evaluate(net, criterion, test_x, test_y)

            loss_gap = test_loss - train_loss
            acc_gap = test_acc - train_acc

            weights_tensor = torch.stack(weights_tensor, requires_grad=False)
            weights_np = weights_tensor.cpu().numpy()
            losses_np = torch.stack(loss_history).cpu().numpy()

            # jump_size is a parameter of the persistent homology part
            # it defines how many finite set are drawn to perform the affine regression
            jump_size = int((ripser_points - min_points) / jump)

            logger.info("Computing euclidean PH dim...")
            ph_dim_euclidean = fast_ripser(weights_np,
                                           max_points=ripser_points,
                                           min_points=min_points,
                                           point_jump=jump_size)

            logger.info("Computing L1 losses based PH dim...")
            ph_dim_losses_based = fast_ripser(losses_np,
                                              max_points=ripser_points,
                                              min_points=min_points,
                                              point_jump=jump_size,
                                              metric="manhattan")

            weight_norm = np.linalg.norm(weights_np[-1]).item()

            exp_dict = {
                "ph_dim_euclidean": ph_dim_euclidean,
                "ph_dim_losses_based": ph_dim_losses_based,
                "weight_norm": weight_norm,
                "train_acc": train_acc,
                "eval_acc": test_acc,
                "acc_gap": acc_gap,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "loss_gap": loss_gap,
                "learning_rate": lr,
                "batch_size": int(batch_size_train),
                "LB_ratio": lr / batch_size_train,
                "depth": depth,
                "width": width,
                "model": model,
                "iterations": i,
                "seed": seed,
                "dataset": dataset,
                "init": 'adversarial' if adversarial_init else 'random',
            }

    return exp_dict
