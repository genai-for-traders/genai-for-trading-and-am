from genai4t.model.core import BaseLightningModule
from torch import nn
import torch
from genai4t.model.layers import TemporalConvBlock
from genai4t.forecasting.predictors.util import RobustScaler, BaseEstimator
from typing import Optional, Dict, Any

class WavenetModel(BaseLightningModule):
    """A WaveNet-based model for time series forecasting.
    
    This model implements a WaveNet architecture, which uses dilated causal convolutions
    to capture long-range dependencies in time series data. 
    
    Attributes:
        context_length (int): Number of time steps used as input for prediction.
        prediction_length (int): Number of time steps to predict into the future.
        hidden_dim (int): Dimension of the hidden layers in the network.
        input_dim (int): Dimension of the input features. Defaults to 1.
        kernel_size (int): Size of the convolutional kernel. Defaults to 2.
        dilation_base (int): Base for the exponential dilation rate. Defaults to 2.
        n_blocks (int): Number of WaveNet blocks. Defaults to 6.
        layers_per_block (int): Number of layers in each block. Defaults to 1.
        output_dim (int): Dimension of the output. Defaults to 1.
        lr (float): Learning rate for the optimizer. Defaults to 1e-3.
        weight_decay (float): Weight decay for the optimizer. Defaults to 0.
    """
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        hidden_dim: int,
        input_dim: int = 1,
        kernel_size: int = 2,
        dilation_base: int = 2,
        n_blocks: int = 6,
        layers_per_block: int = 1,
        output_dim: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()
        
        init_block = TemporalConvBlock(
            input_dim,
            hidden_dim,
            kernel_size=1,
            padding=0,
            dilation=1,
            n_layers=layers_per_block)
        
        modules = [init_block]

        for i in range(n_blocks):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation
            block = TemporalConvBlock(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                n_layers=layers_per_block)
            
            modules.append(block)

        self.blocks = nn.Sequential(*modules)

        decoder_modules = [
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
            nn.ReLU(),
            nn.Linear(context_length, prediction_length)]
        
        self.decoder = nn.Sequential(
            *decoder_modules
        )
        
        self._loss = nn.MSELoss()
        self.prediction_length = prediction_length
        self.scaler = RobustScaler(keepdim=True)
        self.context_length = context_length
        
    
    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            past_target: Input tensor of shape [batch_size, context_length] containing
                        the past time series values.
        
        Returns:
            Tensor of shape [batch_size, 1, prediction_length] containing the predicted
            future values.
        """
        scaled_past_target, loc, scale = self.scaler(past_target)
        x: torch.Tensor = scaled_past_target[:, torch.newaxis]
        x = self.blocks(x)
        scaled_y: torch.Tensor = self.decoder(x)[:, 0]
        y = scaled_y * scale + loc
        return y[:, torch.newaxis] 

    def step(self, batch):
        """Performs a single training/validation step.
        
        Args:
            batch: Dictionary containing 'past_target' and 'future_target' tensors.
        
        Returns:
            The computed loss value.
        """
        past_target = batch['past_target']
        future_target = batch['future_target'][:, torch.newaxis]
        yhat = self.forward(past_target)
        assert yhat.shape == future_target.shape
        loss = self._loss(yhat, future_target)
        return loss
    

class WavenetEstimator(BaseEstimator):
    """Estimator class for the WaveNet model.
    
    This class handles the training and evaluation of the WaveNet model.
    
    Attributes:
        prediction_length (int): Number of time steps to predict into the future.
        context_length (int): Number of time steps used as input for prediction.
        hidden_dim (int): Dimension of the hidden layers in the network.
        num_steps (int): Number of training steps.
        input_dim (int): Dimension of the input features. Defaults to 1.
        kernel_size (int): Size of the convolutional kernel. Defaults to 2.
        dilation_base (int): Base for the exponential dilation rate. Defaults to 2.
        n_blocks (int): Number of WaveNet blocks. Defaults to 6.
        layers_per_block (int): Number of layers in each block. Defaults to 1.
        output_dim (int): Dimension of the output. Defaults to 1.
        lr (float): Learning rate for the optimizer. Defaults to 1e-3.
        weight_decay (float): Weight decay for the optimizer. Defaults to 0.
        batch_size (int): Size of the training batches. Defaults to 32.
        num_batches_per_epoch (int): Number of batches per epoch. Defaults to 50.
        trainer_kwargs (Optional[Dict[str, Any]]): Additional arguments for the trainer.
    """
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dim: int,
        num_steps: int,
        kernel_size: int ,
        dilation_base: int,
        n_blocks: int,
        layers_per_block: int,
        lr: float,
        weight_decay: float,
        batch_size: int,
        num_batches_per_epoch: int,
        output_dim: int = 1,
        input_dim: int = 1,
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
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.n_blocks = n_blocks
        self.layers_per_block = layers_per_block
        self.output_dim = output_dim
        self.lr = lr
        self.weight_decay = weight_decay

    def create_lightning_module(self) -> WavenetModel:
        """Creates and returns a new instance of the WaveNet model.
        
        Returns:
            A new instance of WavenetModel configured with the estimator's parameters.
        """
        return WavenetModel(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            hidden_dim=self.hidden_dim,
            input_dim=self.input_dim,
            kernel_size=self.kernel_size,
            dilation_base=self.dilation_base,
            n_blocks=self.n_blocks,
            layers_per_block=self.layers_per_block,
            output_dim=self.output_dim,
            lr=self.lr,
            weight_decay=self.weight_decay
        )