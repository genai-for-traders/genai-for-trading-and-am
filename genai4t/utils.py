from typing import Dict, Any
import yaml
import random
from lightning import seed_everything
import numpy as np
import torch

def load_yml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def set_random_state(random_state: str):
    np.random.seed(random_state)
    random.seed(random_state)
    seed_everything(random_state)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0") 
    elif torch.mps.is_available():
        return torch.device("mps")
    return torch.device('cpu')