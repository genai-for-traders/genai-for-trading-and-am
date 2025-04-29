from torch import nn
import torch
from genai4t.model.layers import TemporalConvBlock


class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) implementation.
    
    A TCN is a type of neural network architecture that uses dilated convolutions to process
    sequential data. It's particularly effective for time series data due to its ability to
    capture long-range dependencies while maintaining a fixed number of parameters.
    
    Attributes:
        blocks (nn.Sequential): Sequence of temporal convolutional blocks
        decoder (nn.Sequential): Final decoder layer with optional activation function
    
    Args:
        input_dim (int): Number of input channels/features
        hidden_dim (int): Number of hidden channels in the convolutional blocks
        output_dim (int): Number of output channels/features
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 2.
        dilation_base (int, optional): Base for exponential dilation rate increase. Defaults to 2.
        n_blocks (int, optional): Number of temporal convolutional blocks. Defaults to 6.
        layers_per_block (int, optional): Number of convolutional layers per block. Defaults to 2.
        act_fn (nn.Module, optional): Activation function for the final layer. Defaults to None.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        kernel_size: int = 2,
        dilation_base: int = 2,
        n_blocks: int = 6,
        layers_per_block: int = 2,
        act_fn: nn.Module = None,
    ):
        super().__init__()
        
        init_block = TemporalConvBlock(
            input_dim,
            hidden_dim,
            kernel_size=1,
            padding=0,
            dilation=1,
            n_layers=layers_per_block)
        
        # Create sequence of temporal convolutional blocks with increasing dilation rates
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

        # Final decoder layer to project to output dimension
        decoder_modules = [nn.Conv1d(hidden_dim, output_dim, kernel_size=1)]
        
        if act_fn is not None:
            decoder_modules.append(act_fn)
        self.decoder = nn.Sequential(*decoder_modules)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim, sequence_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim, sequence_length)
        """
        x = self.blocks(x)
        x = self.decoder(x)
        return x
    
