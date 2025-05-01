import numpy as np
import pandas as pd
from datasets import Dataset


def split_timeseries_with_length(
        ts: np.ndarray,
        length: int) -> np.ndarray:
    """
    Splits a time series array into overlapping windows of a specified length.

    Parameters
    ----------
    ts : np.ndarray
        The input time series array to be split.
    length : int
        The length of each window.

    Returns
    -------
    np.ndarray
        A stacked array of shape (num_windows, length, ...) containing the windows.
    """
    output = [ts[i: i+length] for i in range(len(ts) - length + 1)]
    return np.stack(output)


def dataset_to_pandas(dataset: Dataset) -> pd.DataFrame:
    """
    Converts a HuggingFace Dataset object to a pandas DataFrame, setting the index to 'timestamp' if present and casting all data to float32.

    Parameters
    ----------
    dataset : Dataset
        The HuggingFace Dataset to convert.

    Returns
    -------
    pd.DataFrame
        The resulting DataFrame with 'timestamp' as index (if present) and all data as float32.
    """
    pd_data = dataset.to_pandas()
    if 'timestamp' in pd_data.columns:
        pd_data.set_index('timestamp', inplace=True)
    pd_data = pd_data.astype(np.float32)
    return pd_data