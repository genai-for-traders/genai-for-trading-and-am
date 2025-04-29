import pandas as pd
from gluonts.model.forecast import SampleForecast
from typing import List
import numpy as np
import matplotlib.ticker as mtick
from genai4t.evaluation.metrics import get_cumulative_returns
from matplotlib import pyplot as plt
import seaborn as sns
from genai4t.plot_style import plot_style

def create_predictions_dataframe(
        target: pd.DataFrame,
        forecasts: List[SampleForecast]) -> pd.DataFrame:
    """
    get a dataframe with target, predictions from a list of forecasts 
    and the original target
    """
    long_target = (
        target
        .reset_index()
        .melt(id_vars=['timestamp'], var_name="item_id", value_name='target')
                )

    _predictions = []

    for forecast in forecasts:
        avg_forecast: np.ndarray = np.median(forecast.samples, axis=0)
        prediction_length = avg_forecast.shape[0]
        timestamps = pd.period_range(
            start=forecast.start_date,
            periods=prediction_length,
            freq=forecast.start_date.freq).to_timestamp()
        item_dataframe = pd.DataFrame(
            {
                "item_id": forecast.item_id,
                "timestamp": timestamps,
                "forecast": avg_forecast,
                "horizon": np.arange(1, prediction_length + 1)
            }
        )
        _predictions.append(item_dataframe)
        
    pd_predictions = (
        pd.concat(_predictions)
        .merge(long_target, on=['timestamp', 'item_id'])
        .sort_values(by=['item_id', 'horizon', 'timestamp'])
        .set_index("timestamp")
    )

    return pd_predictions

def plot_cumulative_returns(pd_predictions: pd.DataFrame):
    pd_predictions['cum_return'] = (
        pd_predictions
        .groupby(['item_id'], group_keys=False)
        .apply(lambda df: get_cumulative_returns(df['target'], df['forecast']))
    )
    f, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(
        x="timestamp",
        y='cum_return',
        hue='item_id',
        data=pd_predictions.reset_index(),
        ax=ax)
    # Horizontal reference line at 0%
    ax.axhline(0., color='red', linestyle='--', linewidth=1.2)
    # Format y-axis as percentages (assuming cum_return is in decimal form)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # Title and labels
    ax.set_title("Cumulative Return Over Time by Asset")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative Return")
    # Apply custom styling
    plot_style.apply_plot_style(ax)
    plot_style.apply_grid(ax, axis="y")
    # Handle legend (optional: move outside if too crowded)
    ax.legend(loc="best")