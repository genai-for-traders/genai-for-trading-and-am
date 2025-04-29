from genai4t.model.core import BaseLightningModule
from torch import nn
import lightning as L
import torch
from genai4t.forecasting.predictors.util import RobustScaler, BaseEstimator
from gluonts.torch.modules.loss import NegativeLogLikelihood
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from typing import Optional, Dict, Any

class ProbabilisticNeuralARModel(BaseLightningModule):
    """A probabilistic autoregressive model for time series forecasting.
    
    This model uses a neural network to learn the parameters of a Student's t-distribution
    for making probabilistic predictions. It combines the power of deep learning with
    probabilistic forecasting capabilities.
    
    Attributes:
        context_length (int): Number of time steps used as input for prediction
        prediction_length (int): Number of time steps to predict
        hidden_dim (int): Dimension of the hidden layers in the neural network
        lr (float): Learning rate for optimization
        weight_decay (float): Weight decay (L2 penalty) for optimization
    """
    
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        hidden_dim: int,
        lr: float = 1e-3,
        weight_decay: float = 0.):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()
        
        self.mlp = nn.Sequential(
            nn.Linear(context_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_length * hidden_dim)
        )
        self.prediction_length = prediction_length
        self.scaler = RobustScaler(keepdim=True)
        self.context_length = context_length
        self.distr_output = StudentTOutput()
        self.param_proj = self.distr_output.get_args_proj(hidden_dim)
        self.hidden_dim = hidden_dim
        self.log_loss = NegativeLogLikelihood()
    
    
    def get_parameters(
        self,
        past_target: torch.Tensor
    ) -> DistributionOutput:
        """Get the distribution parameters for the given past target values.
        
        Args:
            past_target: Past target values of shape [batch_size, context_length]
            
        Returns:
            A distribution object representing the predicted distribution
        """
        scaled_past_target, loc, scale = self.scaler(past_target)
        state: torch.Tensor = self.mlp(scaled_past_target).view(-1, self.prediction_length, self.hidden_dim)
        params = self.param_proj(state)
        dist = self.distr_output.distribution(params, loc, scale)
        return dist
    
    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            past_target: Past target values of shape [batch_size, context_length]
            
        Returns:
            A sample from the predicted distribution
        """
        dist = self.get_parameters(past_target)
        output = dist.sample()[:, torch.newaxis]
        return output

    def step(self, batch):
        """Compute the loss for a training step.
        
        Args:
            batch: Dictionary containing 'past_target' and 'future_target' tensors
            
        Returns:
            The mean loss value
        """
        past_target = batch['past_target']
        future_target = batch['future_target']
        dist = self.get_parameters(past_target)
        loss = self.log_loss(dist, future_target)
        return loss.mean()


class ProbabilisticNeuralAREstimator(BaseEstimator):
    """Estimator class for the ProbabilisticNeuralARModel.
    
    This class handles the training and configuration of the ProbabilisticNeuralARModel.
    
    Attributes:
        prediction_length (int): Number of time steps to predict
        context_length (int): Number of time steps used as input for prediction
        hidden_dim (int): Dimension of the hidden layers in the neural network
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
            trainer_kwargs=trainer_kwargs
        )
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay

    def create_lightning_module(self) -> ProbabilisticNeuralARModel:
        """Create and return a new instance of ProbabilisticNeuralARModel.
        
        Returns:
            A new instance of ProbabilisticNeuralARModel configured with the estimator's parameters
        """
        return ProbabilisticNeuralARModel(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
