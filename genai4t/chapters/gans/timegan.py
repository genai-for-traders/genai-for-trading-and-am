from torch import nn
import torch
import itertools
import torch
from torch import nn
from genai4t.chapters.gans.core import BaseGanModule, BaseLightningModule
from genai4t.model.utils import SampleTimeSeries

MSELoss = nn.MSELoss()


def compute_supervisor_loss(h_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Compute the supervisor loss for temporal dynamics prediction.

    Args:
        h_hat: Predicted hidden states
        h: True hidden states

    Returns:
        torch.Tensor: Mean squared error loss between predicted and true next states
    """
    return MSELoss(h_hat[:, :-1, :], h[:, 1:, :])


class PretrainAEModule(BaseLightningModule):
    """
    Autoencoder module for pretraining the embedding network.

    This module pretrains the encoder and decoder to learn meaningful temporal embeddings
    of the input time series data.

    Attributes:
        encoder (nn.Module): Neural network for encoding time series data
        decoder (nn.Module): Neural network for decoding embeddings back to time series
        lr (float): Learning rate for optimization
        weight_decay (float): Weight decay for optimization
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        self.encoder = encoder
        self.decoder = decoder

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            x: Input time series data

        Returns:
            torch.Tensor: Reconstruction loss
        """
        h = self.encoder(x)
        reconstructed_x = self.decoder(h)

        reconst_loss = MSELoss(reconstructed_x, x)
        return reconst_loss


class PretrainSupervisorModule(BaseLightningModule):
    """
    Supervisor module for pretraining temporal dynamics.

    This module pretrains the supervisor network to learn the temporal dynamics
    of the embedded time series data.

    Attributes:
        encoder (nn.Module): Fixed encoder network
        supervisor (nn.Module): Network for learning temporal dynamics
        lr (float): Learning rate for optimization
        weight_decay (float): Weight decay for optimization
    """

    def __init__(
        self,
        encoder: nn.Module,
        supervisor: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        self.encoder = encoder
        self.supervisor = supervisor

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            x: Input time series data

        Returns:
            torch.Tensor: Supervisor loss
        """
        h = self.encoder(x)
        h_supervised = self.supervisor(h)
        loss = compute_supervisor_loss(h_supervised, h)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer for training.

        Only the supervisor parameters are optimized, while the encoder remains fixed.

        Returns:
            torch.optim.Optimizer: Adam optimizer for the supervisor
        """
        optimizer = torch.optim.Adam(
            self.supervisor.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


def get_generator_moment_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    Compute moment loss to preserve statistical properties of generated data.

    This loss ensures that the generated time series maintains similar statistical
    properties (mean and variance) as the real data.

    Args:
        y_true: Real time series data
        y_pred: Generated time series data

    Returns:
        torch.Tensor: Combined loss of mean and variance differences
    """
    y_true_mean = torch.mean(y_true, dim=0)
    y_true_var = torch.var(y_true, dim=0, unbiased=False)

    y_pred_mean = torch.mean(y_pred, dim=0)
    y_pred_var = torch.var(y_pred, dim=0, unbiased=False)

    # Compute mean absolute difference between means
    g_loss_mean = torch.mean(torch.abs(y_true_mean - y_pred_mean))

    # Compute mean absolute difference between standard deviations
    g_loss_var = torch.mean(
        torch.abs(torch.sqrt(y_true_var + 1e-6) - torch.sqrt(y_pred_var + 1e-6))
    )

    return g_loss_mean + g_loss_var


class TimeGanModule(BaseGanModule):
    """
    Main TimeGAN module implementing the complete training framework.

    This class combines all components (autoencoder, supervisor, generator, discriminator)
    and implements the complete TimeGAN training procedure.

    Attributes:
        encoder (nn.Module): Network for encoding time series data
        decoder (nn.Module): Network for decoding embeddings
        supervisor (nn.Module): Network for learning temporal dynamics
        generator (nn.Module): Network for generating synthetic data
        discriminator (nn.Module): Network for discriminating real vs. synthetic data
        sampler (SampleTimeSeries): Utility for sampling noise vectors
        gamma (float): Weight for discriminator loss components
        lr (float): Learning rate for optimization
        weight_decay (float): Weight decay for optimization
        discriminator_steps (int): Number of discriminator steps per training iteration
        generator_steps (int): Number of generator steps per training iteration
        discriminator_warmup (int): Number of warmup steps for discriminator
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        supervisor: nn.Module,
        generator: nn.Module,
        discriminator: nn.Module,
        sampler: SampleTimeSeries,
        gamma: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        discriminator_steps: int = 5,
        generator_steps: int = 5,
        discriminator_warmup: int = 50,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            discriminator_steps=discriminator_steps,
            generator_steps=generator_steps,
            discriminator_warmup=discriminator_warmup,
        )

        self.encoder = encoder
        self.decoder = decoder
        self.supervisor = supervisor
        self.generator = generator
        self.discriminator = discriminator
        self.sampler = sampler

        self._bce_loss = nn.BCELoss()
        self.gamma = gamma

    def configure_optimizers(self):
        generator_opt = torch.optim.Adam(
            itertools.chain(self.generator.parameters(), self.supervisor.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        autoencoder_opt = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        discriminator_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return [generator_opt, autoencoder_opt, discriminator_opt], []

    def discriminator_step(
        self, x: torch.Tensor, z: torch.Tensor, discriminator_opt: torch.optim.Optimizer
    ) -> torch.Tensor:
        """
        Perform a single discriminator training step.

        Args:
            x: Real time series data
            z: Random noise vector
            discriminator_opt: Discriminator optimizer

        Returns:
            torch.Tensor: Discriminator loss
        """
        self.toggle_optimizer(discriminator_opt)

        h = self.encoder(x)
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)

        y_fake = self.discriminator(h_hat)
        y_fake_e = self.discriminator(e_hat)
        y_real = self.discriminator(h)

        d_real_loss = self._bce_loss(y_real, torch.ones_like(y_real))
        d_fake_loss = self._bce_loss(y_fake, torch.zeros_like(y_fake))
        d_fake_e_loss = self._bce_loss(y_fake_e, torch.zeros_like(y_fake_e))
        d_loss = d_real_loss + d_fake_loss + self.gamma * d_fake_e_loss

        # backward
        self.manual_backward(d_loss)
        # update parameters
        discriminator_opt.step()
        # reset grads
        discriminator_opt.zero_grad()
        # unset optimizer
        self.untoggle_optimizer(discriminator_opt)

        return d_loss

    def generator_step(
        self, x: torch.Tensor, z: torch.Tensor, generator_opt: torch.optim.Optimizer
    ) -> torch.Tensor:
        """
        Perform a single generator training step.

        Args:
            x: Real time series data
            z: Random noise vector
            generator_opt: Generator optimizer

        Returns:
            torch.Tensor: Generator loss
        """
        self.toggle_optimizer(generator_opt)

        # FIXED
        h = self.encoder(x)

        # generator, supervisor
        h_hat_supervised = self.supervisor(h)

        # FIXED
        generator_loss_supervised = compute_supervisor_loss(h_hat_supervised, h)

        e_hat = self.generator(z)
        y_fake_e = self.discriminator(e_hat)
        # adversiral loss
        generator_loss_unsupervised_e = self._bce_loss(
            y_fake_e, torch.ones_like(y_fake_e)
        )

        h_hat = self.supervisor(e_hat)
        y_fake = self.discriminator(h_hat)
        # adversiral loss
        generator_loss_unsupervised = self._bce_loss(y_fake, torch.ones_like(y_fake))

        # ????
        x_hat = self.decoder(h_hat)
        generator_moment_loss = get_generator_moment_loss(x, x_hat)

        generator_loss = (
            generator_loss_unsupervised
            + generator_loss_unsupervised_e
            + 100 * torch.sqrt(generator_loss_supervised)
            + 100 * generator_moment_loss
        )

        self.manual_backward(generator_loss)
        # update parameters
        generator_opt.step()
        # zero grad
        generator_opt.zero_grad()
        # untoggle
        self.untoggle_optimizer(generator_opt)
        return generator_loss

    def ae_step(
        self, x: torch.Tensor, autoencoder_opt: torch.optim.Optimizer
    ) -> torch.Tensor:
        """
        Perform a single autoencoder training step.

        Args:
            x: Real time series data
            autoencoder_opt: Autoencoder optimizer

        Returns:
            torch.Tensor: Autoencoder loss
        """
        self.toggle_optimizer(autoencoder_opt)

        # VARIABLE
        h = self.encoder(x)

        # FIXED
        h_supervised = self.supervisor(h)

        # VARIABLE
        reconstructed_x = self.decoder(h)

        # FIXED
        supervisor_loss_supervised = compute_supervisor_loss(h_supervised, h)

        # RECONSTRUCTION
        embedding_loss_t0 = MSELoss(reconstructed_x, x)

        e_loss = 10 * torch.sqrt(embedding_loss_t0) + 0.1 * supervisor_loss_supervised

        self.manual_backward(e_loss)
        # update parameters
        autoencoder_opt.step()
        # zero grad
        autoencoder_opt.zero_grad()
        # untoggle
        self.untoggle_optimizer(autoencoder_opt)

        return e_loss

    def step(
        self,
        x: torch.Tensor,
        train_discriminator: bool = True,
        train_generator: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a complete training step.

        Args:
            x: Real time series data
            train_discriminator: Whether to train the discriminator
            train_generator: Whether to train the generator

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Generator loss and discriminator loss
        """
        generator_opt, autoencoder_opt, discriminator_opt = self.optimizers()
        z = self.sampler.sample(x.size(0)).to(x.device)

        generator_loss = discriminator_loss = None
        if train_generator:
            gen_loss = self.generator_step(x=x, z=z, generator_opt=generator_opt)
            ae_loss = self.ae_step(x, autoencoder_opt=autoencoder_opt)
            generator_loss = gen_loss + ae_loss

        if train_discriminator:
            discriminator_loss = self.discriminator_step(
                x=x, z=z, discriminator_opt=discriminator_opt
            )

        return generator_loss, discriminator_loss
