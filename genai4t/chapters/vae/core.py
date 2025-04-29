import torch
import torch.nn as nn
from typing import Tuple, Protocol
import abc
from genai4t.model.core import BaseLightningModule


class VAESampler(Protocol):
    """Protocol defining the interface for VAE sampling strategies.
    
    This protocol ensures that any sampler implementation provides a method to sample
    from the latent space distribution using the mean and log variance parameters.
    """
    @abc.abstractmethod
    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Sample from the latent space distribution.
        
        Args:
            z_mean: Mean of the latent space distribution, shape (batch_size, latent_dim)
            z_log_var: Log variance of the latent space distribution, shape (batch_size, latent_dim)
            
        Returns:
            Sampled latent vector, shape (batch_size, latent_dim)
        """
        raise NotImplementedError


class GaussSampler(VAESampler):
    """Implementation of the VAE sampler using Gaussian distribution.
    
    This sampler implements the reparameterization trick to enable gradient flow
    through the sampling process.
    """
    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Sample from a Gaussian distribution using the reparameterization trick.
        
        Args:
            z_mean: Mean of the latent space distribution, shape (batch_size, latent_dim)
            z_log_var: Log variance of the latent space distribution, shape (batch_size, latent_dim)
            
        Returns:
            Sampled latent vector using z = μ + σ * ε where ε ~ N(0,1)
        """
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class VAEModule(BaseLightningModule):
    """Variational Autoencoder (VAE) implementation using PyTorch Lightning.
    
    This module implements a VAE with configurable encoder, decoder, and sampling strategy.
    It handles both the reconstruction loss and KL divergence loss components.
    
    Attributes:
        encoder: Neural network that maps input to latent space parameters
        decoder: Neural network that maps latent space to reconstructed input
        sampler: Strategy for sampling from the latent space distribution
        reconstruction_wt: Weight for the reconstruction loss component
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        sampler: VAESampler,
        reconstruction_wt: float = 1.,
        lr: float = 1e-3,
        weight_decay: float = 0.,
        ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.reconstruction_wt = reconstruction_wt

    def _compute_reconstruction_loss(self, x: torch.Tensor, reconstructed_x: torch.Tensor) -> torch.Tensor:
        """Compute the mean squared error reconstruction loss.
        
        Args:
            x: Original input tensor
            reconstructed_x: Reconstructed input tensor
            
        Returns:
            Sum of squared differences between input and reconstruction
        """
        err = torch.pow(x - reconstructed_x, 2)
        reconst_loss = torch.sum(err)
        return reconst_loss

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed_x: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the total VAE loss and its components.
        
        The total loss is a weighted sum of reconstruction loss and KL divergence.
        
        Args:
            x: Original input tensor
            reconstructed_x: Reconstructed input tensor
            z_mean: Mean of the latent space distribution
            z_log_var: Log variance of the latent space distribution
            
        Returns:
            Tuple containing:
                - Total loss (weighted reconstruction + KL)
                - Reconstruction loss
                - KL divergence loss
        """
        bs = x.size(0)
        reconstruction_loss = self._compute_reconstruction_loss(x, reconstructed_x) / bs
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / bs
        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single forward pass and compute losses.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple containing:
                - Total loss
                - Reconstruction loss
                - KL divergence loss
        """
        z_mean, z_log_var = self.encoder(x)
        z = self.sampler.sample(z_mean, z_log_var)
        reconstructed_x = self.decoder(z)
        loss, rec_loss, kl_loss = self.compute_loss(x, reconstructed_x, z_mean, z_log_var)
        return loss, rec_loss, kl_loss

    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step.
        
        Args:
            batch: Input batch
            batch_idx: Index of the current batch
            
        Returns:
            Total loss for backpropagation
        """
        loss, rec_loss, kl_loss = self.step(batch)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_rec_loss', rec_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_kl_loss', kl_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss
