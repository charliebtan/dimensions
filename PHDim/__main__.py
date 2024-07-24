import datetime
import json
import os
from pathlib import Path

import fire
import numpy as np
import wandb
from loguru import logger
from pydantic import BaseModel

from PHDim.train_risk_analysis import main as risk_analysis


# Uncomment for wandb logging
def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


class AnalysisOptions(BaseModel):

    """
    All hyperparameters of the experiement are defined here
    """

    iterations: int = 2.5e5  # Maximum authorized number of iterations
    log_weights: bool = True  # Whether we want to save final weights of the experiment
    batch_size_eval: int = 5000  # batch size used for evaluation
    #lrmin: float = 0.005  # minimum learning rate in teh experiment
    #lrmax: float = 0.1  # maximum learning rate in the experiment
    #bs_min: int = 32  # minimum batch size in the experiment
    #bs_max: int = 256  # maximum batch sie in the experiment
    eval_freq: int = 10000  # at which frequency we evaluate the model (training and validation sets)
    dataset: str = "cifar100"  # dataset we use
    data_path: str = "~/data/"  # where to find the data
    model: str = "cnn"  # model, currently supported: ["fc", "alexnet", "vgg", "lenet"]
    save_folder: str = "./results"  # Where to save the results
    depth: int = 5  # depth of the network (for FCNN)
    width: int = 200  # width of the network (for FCNN)
    optim: str = "SGD"  # Optimizer
    min_points: int = 1000  # minimum number of points used to compute the PH dimension
    #num_exp_lr: int = 6  # Number of batch sizes we use
    #num_exp_bs: int = 6  # Number of learning rates we use
    compute_dimensions: bool = True  # whether or not we compute the PH dimensions
    project_name: str = "ph_dim"  # project name for WANDB logging
    initial_weights: str = None  # Initial weights if they exist, always none in our work
    ripser_points: int = 5000  # Maximum number of points used to compute the PH dimension
    #batch_sizes: list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    seed: int = 0
    jump: int = 20  # number of finite sets drawn to compute the PH dimension, see https://arxiv.org/abs/2111.13171v1
    additional_dimensions: bool = False  # whether or not compute the ph dimensions used in the robustness experiment
    data_proportion: float = 1. # Proportion of data to use (between 0 and 1), used for pytests
    widths: list = [2,4,6,8,10,12,14,16,18,20,22,24,28,32,40,48,56,64]  # Widths of the CNN
    cat: int = 0

    def __call__(self):

        lr = 1e-1
        batch_size = 128
        seed = self.seed

        cat = self.cat

        if cat == 0:
            widths = [2,10,18,26,36,52]
        elif cat == 1:
            widths = [4,12,20,28,40,56]
        elif cat == 2:
            widths = [6,14,22,30,44,60]
        elif cat == 3:
            widths = [8,16,24,32,48,64]

        if seed == 1:
            widths = reversed(widths)

        for width in widths:

            # Initial weights should be stored in

            # Uncomment for wandb logging
            reset_wandb_env()
            wandb.init(project=self.project_name, entity='ctan',
                    config=self.dict())

            exp_dict = risk_analysis(
                self.iterations,
                batch_size,
                self.batch_size_eval,
                lr,
                self.eval_freq,
                self.dataset,
                self.data_path,
                self.model,
                self.depth,
                self.width,
                self.optim,
                self.min_points,
                seed,
                f'{self.model}{width}_{self.dataset}_{seed}.pth',
                self.compute_dimensions,
                ripser_points=self.ripser_points,
                jump=self.jump,
                cnn_width=width,
            )

            wandb.log(exp_dict)
            wandb.finish()

if __name__ == "__main__":
    fire.Fire(AnalysisOptions)
