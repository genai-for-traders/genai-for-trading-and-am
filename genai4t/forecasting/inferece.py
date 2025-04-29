from tqdm.auto import tqdm
import numpy as np
from .core import TimeSeriesInterface, TSBatch
import pandas as pd
from typing import Union, Dict, Iterator, Tuple
from genai4t.utils import set_random_state


def get_contexts(
        time_series: pd.Series,
        start_idx: int,
        batch_size: int,
        context_length: int) -> Iterator[Tuple[slice, TSBatch]]:
    assert start_idx >= context_length
    for start_batch_idx in tqdm(range(start_idx, len(time_series), batch_size)):
        end_batch_idx = min(len(time_series), start_batch_idx + batch_size)

        # we are going to predict time (t+1, t+2, ..., t+horizon)
        # we assume we have information up to t, inclusive
        # (t-context_length+1, t-context_length+2, ..., t)
        context = [
            # add 1 to include t
            time_series.iloc[t - context_length + 1: t + 1]
            for t in range(start_batch_idx, end_batch_idx)
            ]
        batch_slice = slice(start_batch_idx - start_idx, end_batch_idx - start_idx)
        yield batch_slice, context



def forecast_model(
        time_series: pd.Series, 
        ts_interface: TimeSeriesInterface,
        context_length: int,
        test_start_date: str,
        random_state: int = None,
        batch_size: int = 64,
        ) -> np.ndarray:
    
    if isinstance(random_state, int):
        set_random_state(random_state)

    start_idx = time_series.index.get_loc(test_start_date)
    n = len(time_series) - start_idx
    forecast = np.zeros((n, ts_interface.num_samples, ts_interface.horizon))

    context_iterator = get_contexts(
        time_series,
        start_idx=start_idx,
        batch_size=batch_size,
        context_length=context_length)
    
    for batch_slice, context in context_iterator:
        batch_forecast = ts_interface.forecast(
            context=context,
        )
        assert batch_forecast.shape == (len(context), ts_interface.num_samples, ts_interface.horizon)
        forecast[batch_slice] = batch_forecast
    return forecast


def inferece_time_series(
        model: TimeSeriesInterface,
        tname: str,
        time_series: pd.Series,
        horizon: int,
        context_length: int,
        test_start_date: str,
        batch_size: int = 64,
        random_state: int = None,
        ) -> pd.DataFrame:
    
    print(f'forecasting: {tname}')
    forecast = forecast_model(
        time_series,
        ts_interface=model,
        context_length=context_length,
        test_start_date=test_start_date,
        batch_size=batch_size,
        random_state=random_state)
    # avg over sample dimensions
    avg_forecast: np.ndarray = forecast.mean(axis=1)

    _predictions = []
    test_ts = time_series.loc[test_start_date: ]
    assert test_ts.shape[0] == avg_forecast.shape[0]

    for h in range(1, horizon + 1):
        assert h > 0
        target = test_ts[h: ]    
        assert np.allclose(target.to_numpy(), test_ts.shift(-h).dropna().to_numpy())
        prediction_for_horizon = avg_forecast[: len(target), h - 1]

        horizon_prediction = pd.DataFrame(
            data=
                {'target': target.to_numpy(),
                'prediction': prediction_for_horizon,
                'horizon': h,
                'tname': tname
                },
            index=target.index)
        
        _predictions.append(horizon_prediction)

    pd_prediction = pd.concat(_predictions, axis=0, ignore_index=True)
    return pd_prediction
        


def inferece_all_time_series(
        data: pd.DataFrame,
        model_mapper_or_model: Union[TimeSeriesInterface, Dict[str, TimeSeriesInterface]],
        horizon: int,
        context_length: int,
        test_start_date: str,
        batch_size: int = 64,
        random_state: int = None,
        ) -> pd.DataFrame:
    

    _predictions = []
    for tname in data.columns:
        time_series = data[tname]

        tmodel = (
            model_mapper_or_model[tname]
            if isinstance(model_mapper_or_model, dict) else
            model_mapper_or_model
        ) 
        tpredictions = inferece_time_series(
            model=tmodel,
            tname=tname,
            time_series=time_series,
            horizon=horizon,
            context_length=context_length,
            test_start_date=test_start_date,
            batch_size=batch_size,
            random_state=random_state)
        
        _predictions.append(tpredictions)
    predictions = pd.concat(_predictions, axis=0)
    return predictions