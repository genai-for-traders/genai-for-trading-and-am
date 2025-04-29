from torch import nn
import torch
from lightning.pytorch.utilities import grad_norm
import abc
from typing import Tuple
from genai4t.model.utils import SampleTimeSeries
from genai4t.model.core import BaseLightningModule


class GanStepManager():
    """Manages the training schedule between generator and discriminator.
    
    This class controls when to train the generator vs discriminator based on the current step
    and configured parameters. It supports a warmup period for the discriminator before
    starting generator training.
    
    Attributes:
        discriminator_steps (int): Number of steps to train discriminator per cycle
        generator_steps (int): Number of steps to train generator per cycle
        discriminator_warmup (int): Number of initial steps to train only discriminator
    """
    
    def __init__(
        self,
        discriminator_steps: int,
        generator_steps: int,
        discriminator_warmup: int):
        self._step = -discriminator_warmup
        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.total_steps = discriminator_steps + generator_steps
        self.train_discriminator = True

    @property 
    def train_generator(self) -> bool:
        """Determine if generator should be trained at current step.
        
        Returns:
            bool: True if generator should be trained, False otherwise
        """
        return self._step >= 0 and (0 <= self._step % self.total_steps < self.generator_steps)

    def step(self) -> None:
        """Advance the training step counter."""
        self._step += 1


class BaseGanModule(BaseLightningModule):
    """Base class for GAN implementations.
    
    This abstract base class provides the core functionality for GAN training,
    including step management and training loop structure. Concrete implementations
    must implement the step() method to define the specific GAN architecture.
    
    Attributes:
        lr (float): Learning rate for optimizers
        weight_decay (float): Weight decay for optimizers
        discriminator_steps (int): Number of discriminator steps per cycle
        generator_steps (int): Number of generator steps per cycle
        discriminator_warmup (int): Number of warmup steps for discriminator
    """
    
    def __init__(self,
        lr: float = 1e-3,
        weight_decay: float = 0.,
        discriminator_steps: int = 5,
        generator_steps: int = 5,
        discriminator_warmup: int = 50,
        ):
        """Initialize the base GAN module.
        
        Args:
            lr: Learning rate for optimizers
            weight_decay: Weight decay for optimizers
            discriminator_steps: Number of discriminator steps per cycle
            generator_steps: Number of generator steps per cycle
            discriminator_warmup: Number of warmup steps for discriminator
        """
        super().__init__(
            lr=lr,
            weight_decay=weight_decay
            )
    
        self.automatic_optimization = False
        self.step_manager = GanStepManager(
            discriminator_steps=discriminator_steps,
            generator_steps=generator_steps,
            discriminator_warmup=discriminator_warmup,
        )

    @abc.abstractmethod
    def step(
        self,
        batch: torch.Tensor,
        train_generator: bool,
        train_discriminator: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single training step.
        
        Args:
            batch: Input batch of real data
            train_generator: Whether to train the generator
            train_discriminator: Whether to train the discriminator
            
        Returns:
            Tuple containing generator loss and discriminator loss
        """
        raise NotImplementedError
    

    def training_step(self, batch, batch_idx) -> None:
        """Perform a training step.
        
        Args:
            batch: Input batch of real data
            batch_idx: Index of the current batch
        """
        train_generator = self.step_manager.train_generator
        train_discriminator: bool = not train_generator
        assert train_generator or train_discriminator
        assert not (train_generator and train_discriminator)

        generator_loss, discriminator_loss = self.step(
            batch,
            train_generator=train_generator,
            train_discriminator=train_discriminator
            )

        if generator_loss is not None:
            self.log('generator_loss', generator_loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        if discriminator_loss is not None:
            self.log('discriminator_loss', discriminator_loss.item(), on_step=True, on_epoch=False, prog_bar=True)

        self.step_manager.step()
    
    def validation_step(self, batch, batch_idx) -> None:
        """Perform a validation step.
        
        Args:
            batch: Input batch of real data
            batch_idx: Index of the current batch
        """
        raise NotImplementedError


class GanModule(BaseGanModule):
    """Standard GAN implementation with generator and discriminator.
    
    This class implements a standard GAN architecture with alternating training
    between generator and discriminator. It uses binary cross entropy loss and
    supports gradient clipping.
    
    Attributes:
        generator (nn.Module): Generator network
        discriminator (nn.Module): Discriminator network
        sampler (SampleTimeSeries): Sampler for generating latent vectors
        lr (float): Learning rate for optimizers
        weight_decay (float): Weight decay for optimizers
        discriminator_steps (int): Number of discriminator steps per cycle
        generator_steps (int): Number of generator steps per cycle
        discriminator_warmup (int): Number of warmup steps for discriminator
        betas (Tuple[float, float]): Beta parameters for Adam optimizer
        clip_gradient_norm (float): Maximum gradient norm for clipping
        log_norm (bool): Whether to log gradient norms
    """
    
    def __init__(
        self,
        generator: nn.Module, 
        discriminator: nn.Module,
        sampler: SampleTimeSeries,
        lr: float = 1e-3,
        weight_decay: float = 0.,
        discriminator_steps: int = 5,
        generator_steps: int = 5,
        discriminator_warmup: int = 50,
        betas: Tuple[float, float] = (0.5, 0.999),
        clip_gradient_norm: float = None,
        log_norm: bool = False,
        ):
        """Initialize the GAN module.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            sampler: Sampler for generating latent vectors
            lr: Learning rate for optimizers
            weight_decay: Weight decay for optimizers
            discriminator_steps: Number of discriminator steps per cycle
            generator_steps: Number of generator steps per cycle
            discriminator_warmup: Number of warmup steps for discriminator
            betas: Beta parameters for Adam optimizer
            clip_gradient_norm: Maximum gradient norm for clipping
            log_norm: Whether to log gradient norms
        """
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            discriminator_steps=discriminator_steps,
            generator_steps=generator_steps,
            discriminator_warmup=discriminator_warmup,
        )
        self.generator = generator
        self.discriminator = discriminator
        self.sampler = sampler
        self._bce_loss = nn.BCEWithLogitsLoss()
        self.betas = betas
        self.clip_gradient_norm = clip_gradient_norm
        self.log_norm = log_norm

    
    def configure_optimizers(self):
        """Configure optimizers for generator and discriminator.
        
        Returns:
            List of optimizers and empty list of schedulers
        """
        generator_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            )

        discriminator_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            )

        return [generator_opt, discriminator_opt], []


    def discriminator_step(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        discriminator_opt: torch.optim.Optimizer) -> torch.Tensor:
        """Perform a single discriminator training step.
        
        Args:
            x: Real data batch
            z: Latent vectors for generator
            discriminator_opt: Discriminator optimizer
            
        Returns:
            Discriminator loss
        """
        self.toggle_optimizer(discriminator_opt)
        try:
            x_fake = self.generator(z)
        
            y_fake = self.discriminator(x_fake)
            y_real = self.discriminator(x)

            d_real_loss = self._bce_loss(y_real, torch.ones_like(y_real))
            d_fake_loss = self._bce_loss(y_fake, torch.zeros_like(y_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2

            discriminator_opt.zero_grad()
            self.manual_backward(d_loss)
            
            if isinstance(self.clip_gradient_norm, float):
                self.clip_gradients(discriminator_opt, self.clip_gradient_norm, gradient_clip_algorithm='norm')

            if self.log_norm:
                norms = grad_norm(self, norm_type=2)
                norms['gen_step'] = 0
                self.log_dict(norms)

            discriminator_opt.step()
        finally:
            self.untoggle_optimizer(discriminator_opt)

        return d_loss
       

    def generator_step(
        self,
        z: torch.Tensor,
        generator_opt: torch.optim.Optimizer) -> torch.Tensor:
        """Perform a single generator training step.
        
        Args:
            z: Latent vectors for generator
            generator_opt: Generator optimizer
            
        Returns:
            Generator loss
        """
        self.toggle_optimizer(generator_opt)
        try:
            x_fake = self.generator(z)
            y_fake = self.discriminator(x_fake)
                
            generator_loss = self._bce_loss(y_fake, torch.ones_like(y_fake))

            # zero grad
            generator_opt.zero_grad()
            self.manual_backward(generator_loss)

             # clip gradients
            if isinstance(self.clip_gradient_norm, float):
                self.clip_gradients(generator_opt, self.clip_gradient_norm, gradient_clip_algorithm='norm')
            
            # TODO: KEEP NORMS? 
            if self.log_norm:
                norms = grad_norm(self, norm_type=2)
                norms['gen_step'] = 0
                self.log_dict(norms)

            # update parameters
            generator_opt.step()
        finally:
            # untoggle
            self.untoggle_optimizer(generator_opt)

        return generator_loss

    def step(
        self,
        x: torch.Tensor,
        train_discriminator: bool = True,
        train_generator: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single training step for both generator and discriminator.
        
        Args:
            x: Real data batch
            train_discriminator: Whether to train discriminator
            train_generator: Whether to train generator
            
        Returns:
            Tuple containing generator loss and discriminator loss
        """
        self.zero_grad()
        generator_opt, discriminator_opt = self.optimizers()
        z = self.sampler.sample(x.size(0)).to(x.device)

        discriminator_loss = generator_loss = None
        if train_generator:
            generator_loss = self.generator_step(z=z, generator_opt=generator_opt)

        if train_discriminator:
            discriminator_loss = self.discriminator_step(x=x, z=z, discriminator_opt=discriminator_opt)
        return generator_loss, discriminator_loss
