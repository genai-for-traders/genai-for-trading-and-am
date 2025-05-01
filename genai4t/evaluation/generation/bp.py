from .predictive import PredictiveEvaluator
from .discriminator import DiscriminatorEvaluator
from .core import evaluate_iterations
from .visualize import visualize_2d_pca, visualize_2d_tsne, visualize_time_series
import numpy as np
import pandas as pd
from typing import Tuple


def evaluate(
    real_data: np.ndarray,
    synthetic_data_iters: np.ndarray,
    random_state: int,
    max_iters: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Predictive
    predictive_eval = PredictiveEvaluator(
        n_feat=real_data.shape[-1], random_state=random_state
    )
    predictive_scores = evaluate_iterations(
        predictive_eval,
        real_data,
        synthetic_data_iters,
        random_state=random_state,
        max_iters=max_iters,
    )
    predictive_scores = predictive_scores.describe().T

    # Classification

    discriminator_eval = DiscriminatorEvaluator(
        n_feat=real_data.shape[-1], random_state=random_state
    )

    discriminator_scores = evaluate_iterations(
        discriminator_eval,
        real_data,
        synthetic_data_iters,
        random_state=random_state,
        max_iters=max_iters,
    )

    discriminator_scores = discriminator_scores.describe().T

    return predictive_scores, discriminator_scores


def visualize(real_data: np.ndarray, synthetic_data: np.ndarray):
    for i in np.random.choice(len(real_data), size=1):
        visualize_time_series(real_data[i], synthetic_data[i])

    avg_real_data = real_data.mean(axis=2)
    avg_synthetic_data = synthetic_data.mean(axis=2)

    visualize_2d_pca(avg_real_data, avg_synthetic_data)
    visualize_2d_tsne(avg_real_data, avg_synthetic_data)
