import torch
import abc
import lightning as L


class BaseLightningModule(L.LightningModule):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.0):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

    @abc.abstractmethod
    def step(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("valid_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
