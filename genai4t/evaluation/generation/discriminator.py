import numpy as np
from typing import Dict
from torch.utils.data import TensorDataset, DataLoader
import torch
from .core import Evaluator
from ...model.layers import LSTMClassifier
from ...model.utils import fit_model, init_linear_weights
from torch import nn
from .util import create_real_synt_xy
from sklearn.model_selection import train_test_split
import tempfile
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score)
import pandas as pd
from dataclasses import dataclass


def get_discrimination_scores(
        clf: nn.Module,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        num_steps: int,
        random_state: int,
        test_size: int = 0.2,
        batch_size: int = 32,
        plot: bool = True,
        ) -> pd.Series:
    real_and_fake_data, target = create_real_synt_xy(real_data, synthetic_data)
    target = target[:, np.newaxis].astype(np.float32)

    train_disc_data, test_disc_data, train_is_real, test_is_real = train_test_split(
        real_and_fake_data,
        target,
        test_size=test_size,
        stratify=target.ravel().astype(np.int8),
        random_state=random_state
    )

    train_mean = train_is_real.mean()
    test_mean = test_is_real.mean()
    assert np.allclose(train_mean, test_mean, atol=0.01), \
        f'train_mean: {train_mean}, test_mean: {test_mean}'

    train_disc_ds = TensorDataset(torch.from_numpy(train_disc_data), torch.from_numpy(train_is_real))
    train_dl = DataLoader(train_disc_ds, batch_size=batch_size, shuffle=True)

    test_disc_ds = TensorDataset(torch.from_numpy(test_disc_data), torch.from_numpy(test_is_real))
    test_dl = DataLoader(test_disc_ds, batch_size=batch_size, shuffle=False)


    with tempfile.TemporaryDirectory() as tempdir:
        fit_model(
            tempdir,
            model=clf,
            train_dl=train_dl,
            num_steps=num_steps,
            valid_dl=test_dl,
            random_state=random_state,
            plot=plot,
            enable_progress_bar=False)

    with torch.no_grad():
        is_real_pred = torch.concatenate([clf(x) for (x, _) in test_dl])
        is_real_pred = torch.sigmoid(is_real_pred).cpu().numpy()

    is_real_target: np.ndarray = test_is_real.astype(np.int32).ravel()
    bin_is_real_pred = (is_real_pred >= 0.5).astype(np.int32)


    precision, recall, f1, _ = precision_recall_fscore_support(is_real_target, bin_is_real_pred, average='binary')

    clf_scores = {
        'auc': roc_auc_score(test_is_real, is_real_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'acc': accuracy_score(is_real_target, bin_is_real_pred)
    }

    return clf_scores


@dataclass
class DiscriminatorEvaluator(Evaluator):
    n_feat: int
    random_state: int
    num_steps: int = 1_000
    hidden_dim: int = 24
    lr: float = 1e-3


    def build_model(self) -> LSTMClassifier:
        clf = LSTMClassifier(
            n_feat=self.n_feat,
            hidden_dim=self.hidden_dim,
            lr=self.lr).apply(init_linear_weights)
        return clf

    def eval(self, real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
        clf = self.build_model()
        clf_scores = get_discrimination_scores(
            clf=clf,
            real_data=real_data,
            synthetic_data=synthetic_data,
            random_state=self.random_state,
            num_steps=self.num_steps,
            plot=False)

        return clf_scores
