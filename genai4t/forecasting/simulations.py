from typing import Tuple, List
import pandas as pd
import numpy as np
from .core import TimeSeriesInterface
from matplotlib import pyplot as plt
from genai4t.utils import set_random_state

FIGSIZE = (20, 5)

def create_forecast_dataframe(forecast: np.ndarray, timestamp: pd.DatetimeIndex) -> pd.DataFrame:
    # forecast shape: (num_samples, horizon)
    pd_forecast = pd.DataFrame(forecast.T, index=timestamp)
    pd_forecast = pd_forecast.rename_axis('timestamp')
    pd_forecast = (
        pd_forecast
        .reset_index()
        .melt(id_vars=['timestamp'], var_name='horizon', value_name='prediction')
    )
    pd_forecast['horizon'] += 1
    return pd_forecast


def plot_forecast(
        context: pd.Series,
        target: pd.Series,
        forecast: np.ndarray,
        ax: plt = None):
    # context: context length
    # target (horizon)
    # forecast shape: (num_samples, horizon)
    if ax is None:
        _, ax = plt.subplots()
    
    ax.plot(context.index, context, label='context')
    ax.plot(target.index, target, label='target')
    low, median, high = np.quantile(forecast, [0.1, 0.5, 0.9], axis=0)
    ax.plot(target.index, median, color="tomato", label="median forecast")
    ax.fill_between(target.index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
    ax.legend()


def permute_time_series(time_series: pd.Series) -> pd.Series:
    permutation = np.random.permutation(time_series.to_numpy())
    return pd.Series(permutation, index=time_series.index)


def compute_permutation_forecasts(
        time_series: pd.Series,
        npermutations: int,
        model: TimeSeriesInterface,
        seed: int = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
    
    if seed:
        set_random_state(seed)

    permutations = [permute_time_series(time_series) for _ in range(npermutations)]
    permutations.insert(0, time_series)
    forecasts = model.forecast(permutations)
    return permutations, forecasts


def plot_all_permutations(
        permutations: List[pd.Series],
        target: pd.Series,
        permutation_forecasts: np.ndarray):
    for i in range(len(permutations)):
        permutation_context = permutations[i]
        permutation_forecast = permutation_forecasts[i]
        
        _, ax = plt.subplots(figsize=FIGSIZE)
        ax: plt

        plot_forecast(permutation_context, target, permutation_forecast, ax=ax)
        if i == 0:
            title = 'Original Forecast'
        else:
            title = f'{i}th permutation forecast'
        ax.set_title(title)


def plot_original_vs_permutation_forecast(
        timestamps: pd.DatetimeIndex,
        permutation_forecasts: np.ndarray):
    avg_permutation_forecasts = np.median(permutation_forecasts, axis=1)

    originalf = avg_permutation_forecasts[0]
    permutationsf = avg_permutation_forecasts[1:]
    _, ax = plt.subplots(figsize=FIGSIZE)
    ax: plt

    ax.plot(timestamps, originalf, label='original forecast')
    low, median, high = np.quantile(permutationsf, [0.1, 0.5, 0.9], axis=0)
    ax.plot(timestamps, median, color="tomato", label="median forecast")
    ax.fill_between(timestamps, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
    ax.legend()



