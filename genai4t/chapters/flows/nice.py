import torch
from torch import nn
from typing import Tuple
from genai4t.chapters.flows.core import MLPLayer, create_checkboard_mask

class NiceCouplingLayer(nn.Module):
    """
    A coupling layer implementation following the NICE (Non-linear Independent Components Estimation) framework.
    
    This layer implements a bijective transformation that splits the input into two parts and applies
    an affine transformation to one part based on the other. The transformation is designed to be
    easily invertible while allowing for complex non-linear transformations.
    
    Args:
        network (nn.Module): A neural network that computes the transformation parameters
        mask (torch.Tensor): A binary mask that splits the input into two parts
        feat_dim (int): The dimension of the feature space
    """
    def __init__(
        self,
        network: nn.Module,
        mask: torch.Tensor,
        feat_dim: int,
    ):
        super().__init__()
        self.network = network
        self.register_buffer('mask', mask)
        self.scaling_factor = nn.Parameter(torch.zeros(feat_dim))

    def forward(
        self,
        z: torch.Tensor,
        ldj: torch.Tensor,
        reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the coupling layer.
        
        Args:
            z (torch.Tensor): Input tensor to transform
            ldj (torch.Tensor): Log determinant of the Jacobian
            reverse (bool): If True, performs the inverse transformation
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed tensor and updated log determinant
        """
        # compute scale and bias based a section of the input
        z_input = self.mask * z
        bias = self.network(z_input)
        
        output_mask = 1 - self.mask
        bias = bias * output_mask
        
        if not reverse:
            # shift first and then slace, it is a design choice and have no impact on the output
            z_output = z + bias
            ldj += 0
            return z_output, ldj
        else:
            z_inverse = z - bias
            ldj -= 0
            return z_inverse, ldj
        

def create_nice_layer(seq_len: int, feat_dim: int, invert: bool) -> NiceCouplingLayer:
    """
    Creates a NICE coupling layer with a checkerboard mask pattern.
    
    Args:
        seq_len (int): Length of the sequence
        feat_dim (int): Dimension of the feature space
        invert (bool): Whether to invert the checkerboard mask pattern
        
    Returns:
        NiceCouplingLayer: A configured NICE coupling layer
    """
    mask = create_checkboard_mask(seq_len, invert=invert)
    network = MLPLayer(seq_len, feat_dim)
    clayer = NiceCouplingLayer(network, mask, feat_dim)
    return clayer