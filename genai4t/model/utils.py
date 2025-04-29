
import torch
from torch import nn
import lightning as L
from torch.utils.data import DataLoader
from typing import Dict, Any
import os
import shutil
import pandas as pd
from matplotlib import pyplot as plt
from lightning.pytorch.loggers import CSVLogger
from genai4t.plot_style import plot_style

MODEL_FILENAME = 'model.ckpt'

class SampleTimeSeries(nn.Module):
    def __init__(self, seq_len: int, n_feat: int):
        super().__init__()
        self.seq_len = seq_len
        self.n_feat = n_feat

    def sample(self, bs: int):
        sampled_ts = torch.randn(bs, self.seq_len, self.n_feat)
        return sampled_ts


def plot_train_progress(logdir: str, ax: plt = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 5))
    metrics = pd.read_csv(f'{logdir}/lightning_logs/version_0/metrics.csv')
    metrics = metrics.dropna(subset=['train_loss_epoch'])
    # we skip the first epoch because it is normally much larger and the plot 
    # does not good
    metrics.iloc[1: ].plot(x='epoch', y='train_loss_epoch', ax=ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss over time")
    plot_style.apply_grid(ax)
    plot_style.apply_plot_style(ax)

        

def fit_model(
        logdir: str,
        model: nn.Module,
        train_dl: DataLoader,
        num_steps: int,
        reset_logdir: bool = True,
        valid_dl: DataLoader = None,
        plot: bool = True,
        random_state: int = None,
        **trainer_args: Dict[str, Any]) -> L.Trainer:
    
    total_n_params = get_num_params(model)
    print(f'total_n_params: {total_n_params}')
    
    if reset_logdir and os.path.exists(logdir):
        print(f'deleting {logdir}..')
        shutil.rmtree(logdir)

    deterministic = random_state is not None 
    if deterministic:
        L.seed_everything(random_state)
 
    logger = CSVLogger(logdir)
    trainer = L.Trainer(
        max_steps=num_steps, 
        default_root_dir=logdir,
        **trainer_args,
        deterministic=deterministic,
        logger=logger)
    
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    if plot:
        plot_train_progress(logdir)

    # save model to disk
    output_path = os.path.join(logdir, MODEL_FILENAME)
    trainer.save_checkpoint(output_path)
    return trainer


def get_num_params(module: nn.Module) -> int:
    return sum([p.numel() for p in module.parameters()])


def init_linear_weights(m: nn.Module) -> None:
    for layer in m.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


