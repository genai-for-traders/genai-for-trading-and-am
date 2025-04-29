from gluonts.transform import (
    InstanceSplitter,
    TestSplitSampler,
)
from gluonts.dataset.field_names import FieldName
import torch
from gluonts.torch.scaler import Scaler
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    Chain,
    Identity,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
)
from typing import Optional, Dict, Any, Iterable
import lightning as L
from gluonts.dataset.loader import TrainDataLoader, InferenceDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.dataset.common import Dataset


def get_training_splitter(context_length: int, prediction_length: int) -> InstanceSplitter:
    training_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,  # Field containing the time series values
        is_pad_field=FieldName.IS_PAD,  # Field indicating padded values
        start_field=FieldName.START,  # Start time of the time series
        forecast_start_field=FieldName.FORECAST_START,  # Start time of the forecast
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=1,  # Number of samples per time series
            min_future=prediction_length,  # Minimum future length required
        ),
        past_length=context_length,  # Length of past context window
        future_length=prediction_length,  # Length of future prediction window
    )
    return training_splitter


def get_prediction_splitter(
        context_length: int,
        prediction_length: int,
) -> InstanceSplitter:
    prediction_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=context_length,
            future_length=prediction_length,
        )
    return prediction_splitter


class RobustScaler(Scaler):
    """
    Computes a scaling factor by removing the median and scaling by the
    interquartile range (IQR).

    Parameters
    ----------
    dim
        dimension along which to compute the scale
    keepdim
        controls whether to retain dimension ``dim`` (of length 1) in the
        scale tensor, or suppress it.
    minimum_scale
        minimum possible scale that is used for any item.
    """

    def __init__(
        self,
        dim: int = -1,
        keepdim: bool = False,
        minimum_scale: float = 1e-10,
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    def __call__(
        self, data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            observed_data = data

            med = torch.median(observed_data, dim=self.dim, keepdim=True).values
            q1 = torch.quantile(observed_data, 0.25, dim=self.dim, keepdim=True)
            q3 = torch.quantile(observed_data, 0.75, dim=self.dim, keepdim=True)
            iqr = q3 - q1

            # if observed data is all zeros, nanmedian returns nan
            loc = torch.where(torch.isnan(med), torch.zeros_like(med), med)
            scale = torch.where(torch.isnan(iqr), torch.ones_like(iqr), iqr)
            scale = torch.maximum(scale, torch.full_like(iqr, self.minimum_scale))

            scaled_data = (data - loc) / scale

            if not self.keepdim:
                loc = torch.squeeze(loc, dim=self.dim)
                scale = torch.squeeze(scale, dim=self.dim)

            # assert no nans in scaled data, loc or scale
            assert not torch.any(torch.isnan(scaled_data))
            assert not torch.any(torch.isnan(loc))
            assert not torch.any(torch.isnan(scale))
            assert not torch.any(scale == 0)

            return scaled_data, loc, scale


class BaseEstimator(PyTorchLightningEstimator):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        num_steps: int,
        batch_size: int = 128,
        num_batches_per_epoch: int = 100,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not trainer_kwargs:
            trainer_kwargs = {}
        trainer_kwargs['max_steps'] = num_steps
        trainer_kwargs["max_epochs"] = -1
        super().__init__(trainer_kwargs=trainer_kwargs)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        

    def create_transformation(self) -> Transformation:
        # No transformation applied
        return Chain([Identity()])

    def create_predictor(
        self,
        transformation: Transformation,
        trained_model: L.LightningModule,
    ) -> PyTorchPredictor:

        prediction_splitter = get_prediction_splitter(
            context_length=self.context_length,
            prediction_length=self.prediction_length)
        
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=["past_target"],
            prediction_net=trained_model,
            batch_size=self.batch_size,
            input_transform=transformation + prediction_splitter,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: L.LightningModule,
        **kwargs,
    ) -> Iterable:
        training_splitter = get_training_splitter(self.context_length, self.prediction_length)
        # Create the training data loader
        # This handles batching and preprocessing of training data
        train_dl = TrainDataLoader(
            Cached(data),  # Cache dataset for faster training
            batch_size=self.batch_size,  # Number of samples per batch
            stack_fn=batchify,  # Function to combine samples into batches
            transform=training_splitter,  # Preprocessing transformation
            num_batches_per_epoch=self.num_batches_per_epoch,  # Number of batches per training epoch
)
        return train_dl

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: L.LightningModule,
        **kwargs,
    ) -> Iterable:
        # Create the training data loader
        # This handles batching and preprocessing of training data
        training_splitter = get_training_splitter(self.context_length, self.prediction_length)
        valid_dl = InferenceDataLoader(
            Cached(data),  # Cache dataset for faster training
            batch_size=self.batch_size,  # Number of samples per batch
            stack_fn=batchify,  # Function to combine samples into batches
            transform=training_splitter,  # Preprocessing transformation
            num_batches_per_epoch=self.num_batches_per_epoch,  # Number of batches per training epoch
        )

        return valid_dl