from genai4t.model.core import BaseLightningModule
from torch import nn
import torch
from genai4t.forecasting.predictors.util import RobustScaler, BaseEstimator
from typing import Optional, Dict, Any


class NeuralARModel(BaseLightningModule):
    """A point-wise MLP model for time series forecasting.

    This model uses a simple MLP architecture to predict future values based on past observations.
    It includes robust scaling of the input data for normalization.

    Attributes:
        context_length (int): Number of past time steps used for prediction
        prediction_length (int): Number of future time steps to predict
        hidden_dim (int): Dimension of the hidden layer in the MLP
        lr (float): Learning rate for model training
        weight_decay (float): Weight decay (L2 penalty) for regularization
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        hidden_dim: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()

        self.mlp = nn.Sequential(
            nn.Linear(context_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_length),
        )
        self._loss = nn.MSELoss()
        self.prediction_length = prediction_length
        self.scaler = RobustScaler(keepdim=True)
        self.context_length = context_length

    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past_target (torch.Tensor): Input tensor of past time series values
                with shape (batch_size, context_length)

        Returns:
            torch.Tensor: Predicted future values with shape (batch_size, 1, prediction_length)
        """
        scaled_past_target, loc, scale = self.scaler(past_target)
        scaled_y: torch.Tensor = self.mlp(scaled_past_target)
        y: torch.Tensor = scaled_y * scale + loc
        y: torch.Tensor = y[:, torch.newaxis]
        return y

    def step(self, batch):
        """Compute the loss for a single training step.

        Args:
            batch: Dictionary containing 'past_target' and 'future_target' tensors

        Returns:
            torch.Tensor: MSE loss between predictions and ground truth
        """
        past_target = batch["past_target"]
        future_target = batch["future_target"][:, torch.newaxis]
        yhat = self.forward(past_target)
        assert yhat.shape == future_target.shape
        loss = self._loss(yhat, future_target)
        return loss


class NeuralAREstimator(BaseEstimator):
    """Estimator class for training and evaluating NeuralARModel.

    This class handles the training process and model creation for the NeuralARModel.

    Attributes:
        prediction_length (int): Number of future time steps to predict
        context_length (int): Number of past time steps used for prediction
        hidden_dim (int): Dimension of the hidden layer in the MLP
        num_steps (int): Number of training steps
        lr (float): Learning rate for model training
        weight_decay (float): Weight decay for regularization
        batch_size (int): Size of training batches
        num_batches_per_epoch (int): Number of batches per training epoch
        trainer_kwargs (Optional[Dict[str, Any]]): Additional arguments for the PyTorch Lightning trainer
    """

    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dim: int,
        num_steps: int,
        lr: float,
        weight_decay: float,
        batch_size: int,
        num_batches_per_epoch: int,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            num_steps=num_steps,
            trainer_kwargs=trainer_kwargs,
        )
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay

    def create_lightning_module(self) -> NeuralARModel:
        """Create and return a new instance of NeuralARModel with the estimator's configuration.

        Returns:
            NeuralARModel: A new instance of the autoregressive model
        """
        return NeuralARModel(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
