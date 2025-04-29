import torch
from torch import nn 
from gluonts.torch.model.predictor import PyTorchPredictor
from genai4t.forecasting.predictors.util import get_prediction_splitter


class NaiveModel(nn.Module):
    def __init__(
        self,
        context_length: int,
        prediction_length: int):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
    
    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        last_target = past_target[:, [-1]]
        return last_target.repeat(1, self.prediction_length)[:, torch.newaxis]

    def get_predictor(self, batch_size=32):
        prediction_splitter = get_prediction_splitter(
            context_length=self.context_length,
            prediction_length=self.prediction_length)
        
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=["past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=prediction_splitter,
        )
