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
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from models import fc_bhp
from topology import fast_ripser

from PHDim.hausdorff_alpha import estimator_vector_full, estimator_vector_projected


DATA_SEED = 56  # For splitting in test:train


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def get_weights(net) -> torch.Tensor:
    with torch.no_grad():
        w = []

        # TODO: improve this?
        for p in net.parameters():
            w.append(p.view(-1).detach().to(torch.device('cpu')))
        return torch.cat(w)


@torch.no_grad()
def eval_bhp(train_x, train_y, net, obj) -> float:

    net.eval()
    estimated_y = net(train_x)
    assert estimated_y.ndim == 2
    assert estimated_y.shape[1] == 1
    losses = (estimated_y - train_y).pow(2)
    loss = losses.mean().cpu().item()
    return loss, losses.flatten() / train_x.shape[0]


@torch.no_grad()
def eval_non_lipschitz(train_x, train_y, net, obj) -> float:

    net.eval()
    estimated_y = net(train_x)
    assert estimated_y.ndim == 2
    assert estimated_y.shape[1] == 1
    losses = train_x.pow(2).sum(1) * (estimated_y - train_y).pow(2)
    loss = losses.mean().cpu().item()
    return loss, losses.flatten()


VALIDATION_PROPORTION = 0.2
STOPPING_CRITERION = 0.005


class UnknownDatasetError(BaseException):
    ...


class UnknownModelError(BaseException):
    ...


def train_one_model(eval_freq: int = 1000,
                    lr: float = 1.e-3,
                    iterations: int = 100000,
                    width: int = 1000,
                    depth: int = 7,
                    batch_size: int = 32,
                    ripser_points: int = 3000,
                    jump: int = 20,
                    min_points: int = 200,
                    dataset_name: str = "california",
                    model: str = "fcnn",
                    stopping_criterion: float = STOPPING_CRITERION,
                    ph_period: int = None,
                    additional_dimensions: bool = False,
                    save_weights_file = None,
                    seed: int = None,
                    ):

    print(batch_size, lr, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    torch.manual_seed(seed)

    torch.set_float32_matmul_precision('high')

    # data
    if dataset_name == "boston":
        dataset, targets = load_boston(return_X_y=True)
    elif dataset_name == "california":
        dataset, targets = fetch_california_housing(return_X_y=True)
    else:
        raise UnknownDatasetError(f"Dataset {dataset_name} is unknown or not supported")
    dataset = dataset.astype(np.float64)
    targets = targets.astype(np.float64)
    dataset = (dataset - dataset.mean(0)) / dataset.std(0)
    # dataset = (dataset - dataset.mean(0))

    training_set, test_set, training_targets, test_targets = train_test_split(dataset,
                                                                              targets,
                                                                              test_size=VALIDATION_PROPORTION,
                                                                              random_state=DATA_SEED)

    np.random.seed(DATA_SEED)

    # Turning data into tensors
    training_set = torch.from_numpy(training_set.astype(np.float32)).to(device)
    test_set = torch.from_numpy(test_set.astype(np.float32)).to(device)
    training_targets = torch.from_numpy(training_targets.reshape(-1, 1).astype(np.float32)).to(device)
    test_targets = torch.from_numpy(test_targets.reshape(-1, 1).astype(np.float32)).to(device)

    # Defining dataloaders
    dataset_train = torch.utils.data.TensorDataset(training_set, training_targets)
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    def cycle_training(loader):
        while True:
            for data in loader:
                yield data
    cycle_dataloader = cycle_training(dataloader)

    # model
    input_dim = training_set.shape[1]

    if model == "fcnn":
        net = fc_bhp(width=width, depth=depth, input_dim=input_dim).to(device)
    elif model == "attention":
        net = AttentionFCNN(depth=depth, width=width, input_dim=input_dim).to(device)
    else:
        raise UnknownModelError(f"Model {model} not implemented")
    print(net)
    n_weights = 0
    for p in net.parameters():
        n_weights += p.flatten().shape[0]

    # Defining objectives and optimizers
    obj = torch.nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=lr)

    STOP = False
    risk_hist, eval_hist, batch_risk_hist = [], [], []

    weights_history = deque([])
    batch_risk_history = deque([])
    outputs_history = deque([])
    eval_history = deque([])

    # previous_weights = None
    previous_train_loss = None
    CONVERGED = False
    exp_results = {}
    avg_train_loss = 0.

    logger.info("Starting training")
    for i, (x, y) in enumerate(cycle_dataloader):

        with torch.no_grad():

            if i % eval_freq == 0:
                loss_eval, _ = eval_bhp(test_set, test_targets, net, obj)
                eval_hist.append(loss_eval)
                logger.info(f"Evaluation at iteration {i} finished ✅, score (deviation): {np.sqrt(loss_eval)}")

                loss_train, features = eval_bhp(training_set, training_targets, net, obj)
                logger.info(f"Evaluation on training set at iteration {i} finished ✅, score (deviation): {np.sqrt(loss_train)}")

                avg_train_loss += loss_train
                # wandb.log({"training set loss": loss_train})

                # Stopping criterion on instant train loss
                if (i > 0) and (np.abs(loss_train - previous_train_loss) / previous_train_loss < stopping_criterion):
                    if not CONVERGED:
                        logger.info(f"Experiment converged in {i} iterations !!")
                        CONVERGED = True

                previous_train_loss = loss_train

        net.train()

        x, y = x.to(device), y.to(device)

        optim.zero_grad()
        out = net(x)
        loss = obj(out, y)

        if torch.isnan(loss):
            logger.error('Loss has gone nan ❌')
            break

        # calculate the gradients
        loss.backward()

        # take the step
        optim.step()

        # Some logging
        batch_risk_history.append([loss.cpu().item()])
        batch_risk_hist.append([i, loss.cpu().item()])

        if i > iterations:
            CONVERGED = True

        with torch.no_grad():

            if CONVERGED or ((ph_period is not None) and (not CONVERGED) and (i % ph_period == 0)):
                weights_history.append(get_weights(net))
                loss_train, features = eval_bhp(training_set, training_targets, net, obj)
                outputs_history.append(features)
                loss_eval, features = eval_bhp(test_set, test_targets, net, obj)
                eval_history.append(features)

            if len(weights_history) >= ripser_points:
                STOP = True
                if ph_period is not None:
                    weights_history.popleft()
                    outputs_history.popleft()
                    eval_history.popleft()

                # clear cache
        torch.cuda.empty_cache()

        # Saving weights if specified
        if save_weights_file is not None:
            # logger.info(f"Saving last weights in {str(save_weights_file)}")
            torch.save(net.state_dict(), str(save_weights_file))

        if STOP and CONVERGED:

            with torch.no_grad():

                if len(weights_history) < ripser_points:
                    logger.warning("Experiment did not converge")
                    break

                loss_eval, _ = eval_bhp(test_set, test_targets, net, obj)
                eval_hist.append(loss_eval)

                loss_train, _ = eval_bhp(training_set, training_targets, net, obj)
                risk_hist.append([i, loss_train])

                logger.info(f"Final sqrt(losses): train: {round(np.sqrt(loss_train), 2)}, eval: {round(np.sqrt(loss_eval), 2)}")

                weights_history_np = torch.stack(tuple(weights_history)).cpu().numpy()
                outputs_history_np = torch.stack(tuple(outputs_history)).cpu().numpy()
                eval_history_np = torch.stack(tuple(eval_history)).cpu().numpy()

                del weights_history

                jump_size = int((ripser_points - min_points) / jump)

                logger.info("Computing euclidean PH dim...")
                ph_dim_euclidean = fast_ripser(weights_history_np,
                                               max_points=ripser_points,
                                               min_points=min_points,
                                               point_jump=jump_size)

                logger.info("Computing PH dim in output space...")
                ph_dim_losses_based = fast_ripser(outputs_history_np,
                                                  max_points=ripser_points,
                                                  min_points=min_points,
                                                  point_jump=jump_size,
                                                  metric="manhattan")

                logger.debug(f"outputs shape: {outputs_history_np.shape}")

                logger.info("Computing PH dim in eval space...")
                ph_dim_eval_based = fast_ripser(eval_history_np,
                                                max_points=ripser_points,
                                                min_points=min_points,
                                                point_jump=jump_size)

                traj = torch.tensor(weights_history_np, requires_grad=False)

                print(traj.shape)

                alpha_full_5000 = estimator_vector_full(traj)
                alpha_proj_med_5000, alpha_proj_max_5000 = estimator_vector_projected(traj)

                traj_epoch = traj[-len(dataloader):]

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

                exp_dict = {
                    "ph_dim_euclidean": ph_dim_euclidean,
                    "ph_dim_losses_based": ph_dim_losses_based,
                    "ph_dim_eval_based": ph_dim_eval_based,
                    "alpha_full_5000" : alpha_full_5000,
                    "alpha_proj_med_5000": alpha_proj_med_5000,
                    "alpha_proj_max_5000": alpha_proj_max_5000,
                    "alpha_full_epoch": alpha_full_epoch,
                    "alpha_proj_med_epoch": alpha_proj_med_epoch,
                    "alpha_proj_max_epoch": alpha_proj_max_epoch,
                    "std_dist": std_dist,
                    "norm": norm,
                    "step_size": mean_step_size,
                    "train_loss": loss_train,
                    "test_loss": loss_eval,
                    "loss_gap": loss_train - loss_eval,
                    "learning_rate": lr,
                    "batch_size": int(batch_size),
                    "LB_ratio": lr / batch_size,
                    "depth": depth,
                    "width": width,
                    "model": model,
                    "iterations": i,
                    "seed": seed,
                    "dataset": dataset_name,
                }

            break

    return exp_dict


class BHPAnalysis(BaseModel):

    eval_freq: int = 2000
    output_dir: str = "./bhp_experiments"
    iterations: int = 100_000
    save_outputs: bool = False
    project_name: str = "ph_dim"
    width: int = 200
    depth: int = 5
    ripser_points: int = 5000
    jump: int = 20
    min_points: int = 1000
    dataset: str = "california"
    model: str = "fcnn"
    #bs_min: int = 32
    #bs_max: int = 200
    #lr_min: float = 1e-3
    #lr_max: float = 1e-2
    stopping_criterion: float = STOPPING_CRITERION
    ph_period: int = None  # period at which points are taken, if None it will be at the end
    additional_dimensions: bool = False
    seeds: list = [0, 1, 2, 3, 4]
    batch_sizes: list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    learning_rates: list = np.logspace(-4, -1, 10, base=10)

    def __call__(self):

        # Defining the grid of hyperparameters
        lr_tab = self.learning_rates
        bs_tab = self.batch_sizes

        print('lr_tab', lr_tab)
        print('bs_tab', bs_tab)


        for seed in self.seeds:

            for k in range(len(lr_tab)):
                for b in range(len(bs_tab)):

                    reset_wandb_env()
                    wandb.init(project=self.project_name, entity='ctan',
                               config=self.dict()
                    )


                    exp_dict = train_one_model(self.eval_freq,
                                               lr_tab[k],
                                               self.iterations,
                                               self.width,
                                               self.depth,
                                               int(bs_tab[b]),
                                               self.ripser_points,
                                               self.jump,
                                               self.min_points,
                                               self.dataset,
                                               self.model,
                                               self.stopping_criterion,
                                               self.ph_period,
                                               self.additional_dimensions,
                                               f'{self.model}_{self.depth}_{self.dataset}_{lr_tab[k]}_{bs_tab[b]}_{seed}.pth',
                                               seed=seed,
                                               )

                    wandb.log(exp_dict)
                    wandb.finish()

if __name__ == "__main__":
    fire.Fire(BHPAnalysis)
