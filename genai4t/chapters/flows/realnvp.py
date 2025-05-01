import torch
from torch import nn
from typing import Tuple
from genai4t.chapters.flows.core import MLPLayer, create_checkboard_mask


class RealNVPCouplingLayer(nn.Module):
    """A coupling layer implementation for RealNVP (Real Non-Volume Preserving) transformations.

    This layer implements a bijective transformation that splits the input into two parts,
    transforms one part based on the other, and combines them back. The transformation
    is designed to be easily invertible while allowing for complex transformations.

    Args:
        network (nn.Module): A neural network that outputs scale and bias parameters
            for the transformation. The network should output twice the feature dimension
            to provide both scale and bias terms.
        mask (torch.Tensor): A binary mask tensor that determines which features are
            transformed and which are used to compute the transformation parameters.
        feat_dim (int): The dimension of the features being transformed.
    """

    def __init__(
        self,
        network: nn.Module,
        mask: torch.Tensor,
        feat_dim: int,
    ):
        super().__init__()
        self.network = network
        self.register_buffer("mask", mask)
        self.scaling_factor = nn.Parameter(torch.zeros(feat_dim))

    def forward(
        self, z: torch.Tensor, ldj: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the coupling layer transformation.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, seq_len, feat_dim)
            ldj (torch.Tensor): Log determinant of the Jacobian from previous transformations
            reverse (bool): If True, applies the inverse transformation

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed tensor and updated log determinant
            of the Jacobian
        """
        # compute scale and bias based a section of the input
        z_input = self.mask * z
        scale_bias_output = self.network(z_input)
        scale, bias = scale_bias_output.chunk(2, dim=2)

        # avoid large scales
        scaling_factor = self.scaling_factor.exp().view(1, 1, -1)
        scale = scaling_factor * torch.tanh(scale / scaling_factor)

        output_mask = 1 - self.mask
        scale = scale * output_mask
        bias = bias * output_mask

        # we compute exp(scale) to make easier the compution of the log det jacobian
        # which it will just the jacobian
        # moreovoer, we avoid division in the inverse pass by taking exp(-scale)

        if not reverse:
            # shift first and then slace, it is a design choice and have no impact on the output
            z_output = (z + bias) * torch.exp(scale)
            ldj += torch.sum(scale, dim=[1, 2])
            return z_output, ldj
        else:
            z_inverse = (z * torch.exp(-scale)) - bias
            ldj -= torch.sum(scale, dim=[1, 2])
            return z_inverse, ldj


def create_real_nvp_layer(
    seq_len: int, feat_dim: int, invert: bool
) -> RealNVPCouplingLayer:
    """Create a RealNVP coupling layer with a checkerboard mask pattern.

    Args:
        seq_len (int): Length of the sequence
        feat_dim (int): Dimension of the features
        invert (bool): If True, inverts the checkerboard mask pattern

    Returns:
        RealNVPCouplingLayer: A configured coupling layer instance
    """
    mask = create_checkboard_mask(seq_len, invert=invert)
    network = MLPLayer(seq_len, feat_dim)
    clayer = RealNVPCouplingLayer(network, mask, feat_dim)
    return clayer
