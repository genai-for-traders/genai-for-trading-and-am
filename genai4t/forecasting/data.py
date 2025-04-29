import pandas as pd
from gluonts.dataset.split import split  
from gluonts.dataset.pandas import PandasDataset  
from gluonts.dataset.common import Dataset 
from typing import Tuple, Dict, Any
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt
from itertools import islice


def split_train_test_datasets(
        data: pd.DataFrame,
        prediction_length: int,
        test_split_date: str,
        freq: str) -> Tuple[Dataset, Dataset]:
    ds = PandasDataset(dict(data))
    # Split data into training and test sets based on configured split date
    test_period = pd.Period(test_split_date, freq=freq)
    train_ds, test_gen = split(ds, date=test_period)

    # Calculate number of test windows to evaluate the entire test dataset. 
    n_windows = len(data) - data.index.get_loc(test_split_date) - prediction_length
    print(f"number of windows: {n_windows}")
    # Generate test instances with rolling windows
    # Each window will be used to generate one forecast
    test_ds = test_gen.generate_instances(
        prediction_length=prediction_length,  # How far ahead to predict
        windows=n_windows,  # Number of test windows
        distance=1  # Step size between windows
    )
    return train_ds, test_ds 


def highlight_entry(entry: Dict[str, Any], color: str):
    start = entry["start"]
    end = entry["start"] + len(entry["target"])
    plt.axvspan(start, end, facecolor=color, alpha=0.2)


def plot_dataset_splitting(
        original_dataset: Dataset,
        test_pairs: Dataset, 
        figsize=(20, 3),
        grid=True):
    for original_entry in islice(original_dataset, 2):
        name = original_entry['item_id']
        for test_input, test_label in islice(test_pairs, 3):
            plt.figure(figsize=figsize)  # Use the local figsize
            plt.grid(grid)               # Enable grid locally
            to_pandas(original_entry).plot()
            highlight_entry(test_input, "green")
            highlight_entry(test_label, "blue")
            plt.legend([name, "test input", "test target"], loc="upper right")
            plt.title(f"Train test split for {name}")
            plt.show()