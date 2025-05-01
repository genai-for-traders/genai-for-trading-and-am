from gluonts.model.predictor import Predictor
from typing import Optional
from gluonts.dataset.common import Dataset
from gluonts.torch.batchify import batchify
from chronos import ChronosPipeline
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.model.forecast import SampleForecast
from gluonts.dataset.field_names import FieldName
from genai4t.forecasting.predictors.util import get_prediction_splitter


class ChronosPredictor(Predictor):
    """A predictor that uses the Chronos T5 model for time series forecasting.

    Chronos is a language model-based time series forecasting model that uses T5 architecture.
    This predictor wraps the Chronos model to make it compatible with GluonTS's predictor interface.

    Attributes:
        prediction_length (int): Number of time steps to predict into the future.
        pipeline (ChronosPipeline): The underlying Chronos model pipeline.
        lead_time (int): Number of time steps between the end of the context and the start of prediction.
        batch_size (int): Number of samples to process in each batch during inference.
        prediction_splitter: A transform that splits the input data into context and prediction windows.
        field_name (str): The field name used for the target time series data.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        model_name: str,
        device_map: str,
        batch_size: int,
        lead_time: int = 0,
    ):
        self.prediction_length = prediction_length
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device_map,  # use "cpu" for CPU inference and "mps" for Apple Silicon
        )
        self.lead_time = lead_time
        self.batch_size = batch_size

        self.prediction_splitter = get_prediction_splitter(
            context_length=context_length, prediction_length=prediction_length
        )
        self.field_name = "past_target"

    def predict(self, dataset: Dataset, num_samples: Optional[int] = None):
        """Generate forecasts for the input dataset.

        Args:
            dataset (Dataset): The input dataset containing time series data.
            num_samples (Optional[int]): Number of sample paths to generate for each time series.
                If None, uses the default number of samples from the model.

        Yields:
            SampleForecast: Forecast objects containing the predicted samples for each time series
                in the dataset.
        """
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.prediction_splitter,
            batch_size=self.batch_size,
            stack_fn=batchify,
        )

        for batch in inference_data_loader:
            batch_samples = self.pipeline.predict(
                context=batch[self.field_name],
                prediction_length=self.prediction_length,
                num_samples=num_samples,
            )
            batch_samples = batch_samples.cpu().numpy()
            for i, samples in enumerate(batch_samples):
                forecast = SampleForecast(
                    samples,
                    start_date=batch[FieldName.FORECAST_START][i],
                    item_id=(
                        batch[FieldName.ITEM_ID][i]
                        if FieldName.ITEM_ID in batch
                        else None
                    ),
                    info=batch["info"][i] if "info" in batch else None,
                )
                yield forecast
            assert i + 1 == len(batch[FieldName.FORECAST_START])
