import numpy as np
from typing import Tuple


def create_real_synt_xy(
    real_data: np.ndarray, synt_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    test_and_synt_data = np.concatenate((real_data, synt_data))
    is_real = np.zeros(real_data.shape[0] + synt_data.shape[0], dtype=np.int32)
    is_real[: real_data.shape[0]] = 1
    return test_and_synt_data, is_real
