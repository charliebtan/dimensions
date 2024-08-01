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

    iterations: int = 10000000000  # Maximum authorized number of iterations
    log_weights: bool = True  # Whether we want to save final weights of the experiment
    batch_size_eval: int = 5000  # batch size used for evaluation
    lr: float = 1e-2  # minimum learning rate in teh experiment
    bs: int = 128  # minimum batch size in the experiment
    eval_freq: int = 1000  # at which frequency we evaluate the model (training and validation sets)
    dataset: str = "cifar10"  # dataset we use
    data_path: str = "~/data/"  # where to find the data
    model: str = "alexnet"  # model, currently supported: ["fc", "alexnet", "vgg", "lenet"]
    save_folder: str = "./results"  # Where to save the results
    depth: int = 5  # depth of the network (for FCNN)
    width: int = 200  # width of the network (for FCNN)
    optim: str = "SGD"  # Optimizer
    min_points: int = 1000  # minimum number of points used to compute the PH dimension
    num_exp_lr: int = 6  # Number of batch sizes we use
    num_exp_bs: int = 6  # Number of learning rates we use
    compute_dimensions: bool = True  # whether or not we compute the PH dimensions
    project_name: str = "ph_dim"  # project name for WANDB logging
    initial_weights: str = None  # Initial weights if they exist, always none in our work
    ripser_points: int = 5000  # Maximum number of points used to compute the PH dimension
    seeds: list = [i for i in range(10, 30)]
    jump: int = 20  # number of finite sets drawn to compute the PH dimension, see https://arxiv.org/abs/2111.13171v1
    additional_dimensions: bool = False  # whether or not compute the ph dimensions used in the robustness experiment
    data_proportion: float = 1. # Proportion of data to use (between 0 and 1), used for pytests
    random: bool = False  # whether or not to use adversarial initialization

    def __call__(self):

        for seed in self.seeds:

            for random in [False, True]:

                # Initial weights should be stored in

                # Uncomment for wandb logging
                reset_wandb_env()
                wandb.init(project=self.project_name, entity='ctan',
                        config=self.dict(),
                        tags=['adv_init'] if self.random else ['normal_init'],
                        )

                exp_dict = risk_analysis(
                    self.iterations,
                    self.bs,
                    self.batch_size_eval,
                    self.lr,
                    self.eval_freq,
                    self.dataset,
                    self.data_path,
                    self.model,
                    self.depth,
                    self.width,
                    self.optim,
                    self.min_points,
                    seed,
                    f'{self.model}_{self.depth}_{self.dataset}_{self.lr}_{self.bs}_{seed}.pth',
                    self.compute_dimensions,
                    ripser_points=self.ripser_points,
                    jump=self.jump,
                    random=random,
                )

                wandb.log(exp_dict)
                wandb.finish()

if __name__ == "__main__":
    fire.Fire(AnalysisOptions)
