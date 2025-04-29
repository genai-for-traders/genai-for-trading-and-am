from typing import Protocol
import abc
import numpy as np
import pandas as pd
from typing import List

TSBatch = List[pd.Series]


class TimeSeriesInterface(Protocol):
    horizon: int
    num_samples: int

    @abc.abstractmethod
    def forecast(self, context: TSBatch) -> np.ndarray:
        raise NotImplementedError