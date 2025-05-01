import numpy as np
from typing import Tuple, Dict
from torch.utils.data import TensorDataset
import torch
from ...model.layers import LSTMRegressor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import tempfile
from ...model.utils import fit_model, init_linear_weights
from ..metrics import compute_regression_scores
import pandas as pd
from .core import Evaluator
from dataclasses import dataclass


def create_supervised_xy(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    predict last time step
    """
    assert (
        len(data.shape) == 3
    ), f"time series must have 3 dimensions (bs, L, D), got: {len(data.shape)}"
    assert data.shape[1] > 1, "time series must have at least 2 timesteps"

    context = data[:, :-1, :]
    target = data[:, -1, :]

    assert len(context.shape) == 3
    assert len(target.shape) == 2
    return context, target


def create_supervised_dataset(data: np.ndarray) -> TensorDataset:

    assert data.dtype == np.float32
    context, target = create_supervised_xy(data)

    real_train_ds = TensorDataset(torch.from_numpy(context), torch.from_numpy(target))

    return real_train_ds


def compute_predictive_scores(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    reg: LSTMRegressor,
    num_steps: int,
    test_split: float = 0.2,
    random_state: int = 0,
    batch_size: int = 32,
    plot: bool = True,
) -> pd.DataFrame:

    train_idx, test_idx = train_test_split(
        np.arange(len(real_data)), test_size=test_split, random_state=random_state
    )

    real_test_data = real_data[test_idx]

    synt_train_data = synthetic_data[train_idx]

    # real data
    real_test_ds = create_supervised_dataset(real_test_data)
    real_test_dl = DataLoader(real_test_ds, batch_size=batch_size, shuffle=False)

    # synthetic
    sync_train_ds = create_supervised_dataset(synt_train_data)
    synt_train_dl = DataLoader(sync_train_ds, batch_size=batch_size, shuffle=True)

    with tempfile.TemporaryDirectory() as tempdir:
        _ = fit_model(
            logdir=tempdir,
            model=reg,
            train_dl=synt_train_dl,
            num_steps=num_steps,
            plot=plot,
            random_state=random_state,
            enable_progress_bar=False,
        )

    with torch.no_grad():
        yhat = torch.concatenate([reg(x) for (x, _) in real_test_dl]).cpu().numpy()
    real_test_target = real_test_ds.tensors[1].cpu().numpy()
    assert real_test_target.shape == yhat.shape, (real_test_target.shape, yhat.shape)

    flat_target = real_test_target.ravel()
    flat_yhat = yhat.ravel()

    scores = compute_regression_scores(flat_target, flat_yhat)
    return scores


@dataclass
class PredictiveEvaluator(Evaluator):
    n_feat: int
    random_state: int
    num_steps: int = 1_000
    hidden_dim: int = 24
    lr: float = 1e-3

    def build_model(self) -> LSTMRegressor:
        reg = LSTMRegressor(
            n_feat=self.n_feat,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_feat,
            lr=self.lr,
        ).apply(init_linear_weights)
        return reg

    def eval(
        self, real_data: np.ndarray, synthetic_data: np.ndarray
    ) -> Dict[str, float]:
        reg = self.build_model()
        predictive_scores = compute_predictive_scores(
            real_data,
            synthetic_data,
            reg,
            random_state=self.random_state,
            num_steps=self.num_steps,
            plot=False,
        )
        return predictive_scores
