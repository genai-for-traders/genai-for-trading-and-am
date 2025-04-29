from genai4t.model.core import BaseLightningModule
from torch import nn
import torch
from genai4t.forecasting.predictors.util import RobustScaler, BaseEstimator
from typing import Optional, Dict, Any


class LinearARModel(BaseLightningModule):
    """Linear Autoregressive model for time series forecasting.
    
    This class implements a simple linear autoregressive model using PyTorch Lightning.
    It uses a linear layer to predict future values based on past observations.
  
    Attributes:
        context_length (int): Number of past observations used for prediction
        prediction_length (int): Number of future time steps to predict
        linear (nn.Linear): Linear layer for prediction
        _loss (nn.MSELoss): Mean squared error loss function
        scaler (RobustScaler): Robust scaler for input normalization
    """
    
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        lr: float = 1e-3,
        weight_decay: float = 0.):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()
        
        self.linear = nn.Linear(context_length, prediction_length)
        self._loss = nn.MSELoss()
        self.prediction_length = prediction_length
        self.scaler = RobustScaler(keepdim=True)
        self.context_length = context_length
        
    
    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            past_target: Input tensor of shape (batch_size, context_length)
                
        Returns:
            Predicted values of shape (batch_size, 1, prediction_length)
        """
        scaled_past_target, loc, scale = self.scaler(past_target)
        scaled_y: torch.Tensor = self.linear(scaled_past_target)
        y: torch.Tensor = scaled_y * scale + loc
        y: torch.Tensor = y[:, torch.newaxis]
        return y

    def step(self, batch):
        """Compute loss for a single training step.
        
        Args:
            batch: Dictionary containing 'past_target' and 'future_target' tensors
                
        Returns:
            Mean squared error loss
        """
        past_target = batch['past_target']
        future_target = batch['future_target'][:, torch.newaxis]
        yhat = self.forward(past_target)
        assert yhat.shape == future_target.shape
        loss = self._loss(yhat, future_target)
        return loss


class LinearAREstimator(BaseEstimator):
    """Estimator class for the LinearAREstimator.
    
    This class handles the training and configuration of the LinearAREstimator.
    
    Attributes:
        prediction_length (int): Number of time steps to predict
        context_length (int): Number of time steps used as input for prediction
        num_steps (int): Number of training steps
        lr (float): Learning rate for optimization
        weight_decay (float): Weight decay (L2 penalty) for optimization
        batch_size (int): Size of training batches
        num_batches_per_epoch (int): Number of batches per training epoch
        trainer_kwargs (Optional[Dict[str, Any]]): Additional arguments for the trainer
    """
    
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
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
            trainer_kwargs=trainer_kwargs
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def create_lightning_module(self) -> LinearARModel:
        """Create and return a new instance of LinearARModel."""
        return LinearARModel(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            lr=self.lr,
            weight_decay=self.weight_decay
        )