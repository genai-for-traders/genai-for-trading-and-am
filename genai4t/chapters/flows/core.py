import numpy as np
import torch
from torch import nn
from typing import Tuple, List
from genai4t.model import BaseLightningModule


def create_checkboard_mask(seq_len: int, invert: bool = False) -> torch.Tensor:
    """
    Creates a checkerboard pattern mask for alternating transformations in normalizing flows.

    Args:
        seq_len (int): Length of the sequence to create the mask for
        invert (bool, optional): If True, inverts the checkerboard pattern. Defaults to False.

    Returns:
        torch.Tensor: A binary mask tensor of shape (1, seq_len, 1) with alternating 0s and 1s
    """
    indices = torch.arange(seq_len)
    mask = torch.fmod(indices, 2).to(torch.float32).view(1, seq_len, 1)
    if invert:
        mask = 1.0 - mask
    return mask


class TimeSeriesFlowModel(BaseLightningModule):
    """
    A normalizing flow model for time series data.

    This model implements a series of invertible transformations (flows) that can be used
    to transform time series data into a simpler distribution (typically Gaussian) and back.
    The model is trained using maximum likelihood estimation.

    Attributes:
        flows (nn.ModuleList): List of flow transformations to apply
        prior (torch.distributions.Normal): Prior distribution for the latent space
    """

    def __init__(
        self, flows: List[nn.Module], lr: float = 1e-3, weight_decay: float = 0.0
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.flows = nn.ModuleList(flows)
        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes input data into the latent space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - z: Latent representation
                - ldj: Log determinant of the Jacobian
        """
        z = x
        ldj = torch.zeros(x.shape[0], dtype=torch.float32).to(x.device)

        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, x: torch.Tensor, return_ll: bool = False) -> torch.Tensor:
        """
        Computes the negative log-likelihood or log-likelihood of the input data.

        Args:
            x (torch.Tensor): Input tensor
            return_ll (bool, optional): If True, returns log-likelihood instead of bits per dimension.
                                      Defaults to False.

        Returns:
            torch.Tensor: Either bits per dimension (if return_ll=False) or log-likelihood (if return_ll=True)
        """
        z, ldj = self.encode(x)

        log_pz = self.prior.log_prob(z).sum(dim=[1, 2])
        log_px = ldj + log_pz

        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(x.shape[1:])
        return bpd.mean() if not return_ll else log_px

    @torch.no_grad()
    def sample(
        self, shape: List[int], device: torch.device, z_init: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generates samples from the model by sampling from the prior and applying inverse flow transformations.

        Args:
            shape (List[int]): Desired shape of the samples [batch_size, seq_len, feature_dim]
            device (torch.device): Device to place the samples on
            z_init (torch.Tensor, optional): Initial latent representation to transform.
                                           If None, samples from the prior. Defaults to None.

        Returns:
            torch.Tensor: Generated samples of the specified shape
        """
        # Sample latent representation from prior
        if z_init is None:
            z = self.prior.sample(sample_shape=shape).to(device)
        else:
            z = z_init.to(device)
        # Transform z to x by inverting the flows
        ldj = torch.zeros(z.shape[0], device=device)
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z

    def step(self, batch):
        """
        Performs a single training step.

        Args:
            batch: Input batch of data

        Returns:
            torch.Tensor: Training loss (bits per dimension)
        """
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self._get_likelihood(batch)
        return loss


class MLPLayer(nn.Module):
    """
    A simple MLP layer for processing time series data.

    This layer applies a two-layer MLP to the flattened time series data and then
    reshapes it back to the original dimensions.
    """

    def __init__(self, seq_len: int, feat_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(seq_len * feat_dim, seq_len * feat_dim),
            nn.ReLU(),
            nn.Linear(seq_len * feat_dim, seq_len * feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feat_dim)

        Returns:
            torch.Tensor: Processed tensor of the same shape as input
        """
        bs, seq_len, feat_dim = x.shape
        x = x.view(bs, -1)
        x = self.layer(x)
        x = x.view(bs, seq_len, feat_dim)
        return x
