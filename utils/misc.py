
import wandb
import torch
import random
import argparse
import numpy as np

from typing import Dict, Union


def set_random_seed(logger: str, seed: int=42, deterministic=False) -> None:
    """
    Set a random seed for reproducability.

    :param seed: The seed to be used.
    :param deterministic: Whether to set the deterministic option for CUDNN backend.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'Random seed set to: {seed},'
                    f'Deterministic: {deterministic}')
    
def initialize_wandb(args: argparse.Namespace, config: Dict[str, Union[int, float, str]], project_name: str) -> None:
    """
    Initialize Weights and Biases for logging and experiment tracking. 

    :param args: The arguments from the command line. 
    :param config: The configuration dictionary for the experiment.
    :param project_name: The name of the project on wandb.
    """
    wandb.init(project=project_name,
        name=args.exp_name,
        config={**vars(args), **config},
        dir=args.experiment_path)
