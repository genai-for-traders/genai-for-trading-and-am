from genai4t.model.core import BaseLightningModule
from torch import nn
import torch
from genai4t.forecasting.predictors.util import RobustScaler, BaseEstimator
from typing import Optional, Dict, Any

class LSTMModel(BaseLightningModule):
    """A PyTorch Lightning module implementing an LSTM-based time series forecasting model.
    
    This model uses an LSTM network followed by an MLP to predict future values in a time series.
    
    Attributes:
        context_length (int): Number of past time steps used as input for prediction.
        prediction_length (int): Number of future time steps to predict.
        hidden_dim (int): Dimension of the LSTM hidden state.
        num_layers (int): Number of LSTM layers.
        lr (float): Learning rate for optimization.
        weight_decay (float): Weight decay (L2 penalty) for optimization.
    """
    
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        hidden_dim: int,
        num_layers: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 0.):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_length)
        )
        self._loss = nn.MSELoss()
        self.prediction_length = prediction_length
        self.scaler = RobustScaler(keepdim=True)
        self.context_length = context_length
        
    
    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LSTM model.
        
        Args:
            past_target: Input tensor of shape [batch_size, context_length] containing past time series values.
            
        Returns:
            Tensor of shape [batch_size, 1, prediction_length] containing the predicted future values.
        """
        scaled_past_target, loc, scale = self.scaler(past_target)
        scaled_past_target: torch.Tensor = scaled_past_target[..., torch.newaxis]
            
        hidden_states, _ = self.lstm(scaled_past_target)
        last_state: torch.Tensor = hidden_states[:, -1]
        scaled_y = self.mlp(last_state)
        y: torch.Tensor = scaled_y * scale + loc
        y: torch.Tensor = y[:, torch.newaxis]
        return y

    def step(self, batch):
        """Compute the loss for a single training step.
        
        Args:
            batch: Dictionary containing 'past_target' and 'future_target' tensors.
            
        Returns:
            The computed MSE loss between predictions and true values.
        """
        past_target = batch['past_target']
        future_target = batch['future_target'][:, torch.newaxis]
        yhat = self.forward(past_target)
        assert yhat.shape == future_target.shape
        loss = self._loss(yhat, future_target)
        return loss
    

class LSTMEstimator(BaseEstimator):
    """Estimator class for training and using the LSTM forecasting model.
    
    This class handles the creation and training of the LSTM model, providing a high-level
    interface for time series forecasting tasks.
    
    Attributes:
        prediction_length (int): Number of future time steps to predict.
        context_length (int): Number of past time steps used as input for prediction.
        hidden_dim (int): Dimension of the LSTM hidden state.
        num_steps (int): Number of training steps.
        num_layers (int): Number of LSTM layers.
        lr (float): Learning rate for optimization.
        weight_decay (float): Weight decay (L2 penalty) for optimization.
        batch_size (int): Size of training batches.
        num_batches_per_epoch (int): Number of batches per training epoch.
        trainer_kwargs (Optional[Dict[str, Any]]): Additional arguments for the PyTorch Lightning trainer.
    """
    
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dim: int,
        num_steps: int,
        num_layers: int,
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
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay

    def create_lightning_module(self) -> LSTMModel:
        """Create and return an instance of the LSTM model.
        
        Returns:
            An initialized LSTMModel instance with the configured parameters.
        """
        return LSTMModel(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            lr=self.lr,
            weight_decay=self.weight_decay
        )