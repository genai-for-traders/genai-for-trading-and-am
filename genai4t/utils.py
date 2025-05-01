from typing import Dict, Any
import yaml
import random
from lightning import seed_everything
import numpy as np
import torch


def load_yml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file.

    Args:
        path (str): The path to the YAML file.

    Returns:
        Dict[str, Any]: The contents of the YAML file as a dictionary.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def set_random_state(random_state: str):
    """
    Set the random seed for numpy, random, PyTorch, and Lightning to ensure reproducibility.

    Args:
        random_state (str): The seed value to use for all random number generators.
    """
    np.random.seed(random_state)
    random.seed(random_state)
    seed_everything(random_state)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_torch_device() -> torch.device:
    """
    Get the best available torch device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The best available device for torch operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
