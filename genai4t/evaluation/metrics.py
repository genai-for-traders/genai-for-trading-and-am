import numpy as np
from sklearn import metrics
from typing import Dict, Union, List
from gluonts.model.forecast import Forecast 
import pandas as pd
from .simple_evaluator import Evaluator
from scipy.stats import spearmanr

FORECAST_METRIC_NAMES = [
 'MASE',
 'MAPE',
 'sMAPE',
 'RMSE',
 'wQuantileLoss[0.5]',
 'mean_wQuantileLoss'
]


MetricsSummary = Dict[str, Union[float, int]]

def compute_mape(target: np.ndarray, yhat: np.ndarray) -> float:
    pct_error = (target - yhat) / target
    abs_error = np.abs(pct_error)
    return abs_error.mean() * 100


def compute_regression_scores(target: np.ndarray, yhat: np.ndarray) -> MetricsSummary:
    mse = metrics.mean_squared_error(target, yhat)
    mae = metrics.mean_absolute_error(target, yhat)
    mape = compute_mape(target, yhat)

    scores = {
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape
    }

    return scores


def compute_classification_scores(
        labels: np.ndarray,
        yhat: np.ndarray,
        threshold: float = 0.5) -> MetricsSummary:

    bin_yhat = (yhat >= threshold).astype(np.int32)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, bin_yhat, average='binary')

    clf_scores = {
        'auc': metrics.roc_auc_score(labels, yhat),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'acc': metrics.accuracy_score(labels, bin_yhat)
    }

    return clf_scores


def compute_forecast_scores(tss: List[pd.DataFrame], forecasts: List[Forecast]):
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    forecast_metrics_dict, _ = evaluator(tss, forecasts)
    forecast_metrics = pd.DataFrame([forecast_metrics_dict])[FORECAST_METRIC_NAMES]
    return forecast_metrics


def compute_log_cum_returns(log_returns: pd.Series) -> pd.Series:
    log_cum_returns =  np.exp(log_returns.cumsum()) - 1
    return log_cum_returns

def compute_investment_scores(target: pd.Series, yhat: pd.Series) -> MetricsSummary:
    sign = np.sign(yhat)
    returns = target * sign

    r_mean = returns.mean()
    r_std = returns.std()
    r_sharpe = np.sqrt(252) * r_mean / r_std

    cum_return = compute_log_cum_returns(returns).iloc[-1]
    spearman_corr, _ = spearmanr(target, yhat)

    scores = {
        "mean": r_mean,
        "std": r_std,
        "sharpe": r_sharpe,
        "cum_return": cum_return,
        "spearman_corr": spearman_corr
    }
    return scores



def get_cumulative_returns(target: pd.Series, yhat: pd.Series) -> pd.DataFrame:
    sign =  np.sign(yhat)
    returns = target * sign
    cum_return = compute_log_cum_returns(returns)
    return cum_return.to_frame("cum_return")