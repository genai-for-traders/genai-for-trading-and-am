import numpy as np
from typing import Dict, Protocol
import abc
import random
import pandas as pd
from tqdm.auto import tqdm
from ...utils import set_random_state


class Evaluator(Protocol):
    @abc.abstractmethod
    def eval(
        self, real_data: np.ndarray, synthetic_data: np.ndarray
    ) -> Dict[str, float]:
        raise NotImplementedError


def evaluate_iterations(
    evaluator: Evaluator,
    real_data: np.ndarray,
    synthetic_data_iters: np.ndarray,
    random_state: int,
    max_iters: None,
) -> pd.DataFrame:
    scores_db = []

    total_iters = max_iters or len(synthetic_data_iters)
    pbar = tqdm(total=total_iters)

    for i in range(total_iters):
        set_random_state(random_state)
        np.random.seed(random_state)
        random.seed(random_state)

        synthetic_data = synthetic_data_iters[i]

        scores = evaluator.eval(real_data, synthetic_data)

        iter_scores = {
            "iter": i,
            **scores,
        }
        scores_db.append(iter_scores)

        pbar.set_postfix(scores)
        pbar.update()
    pd_scores = pd.DataFrame(scores_db).set_index("iter")
    return pd_scores
