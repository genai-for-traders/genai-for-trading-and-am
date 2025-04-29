from torch import nn
import torch
from typing import Tuple
from .core import BaseLightningModule


class GRULayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        target_dim: int,
        num_layers: int,
        dropout: float = 0.,
        act_fn: nn.Module = nn.Sigmoid()
    ):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.mlp = nn.Linear(hidden_dim, target_dim)

        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(x)
        x = self.mlp(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x



# TODO: We can refactore both in a single class
class LSTMClassifier(BaseLightningModule):
    def __init__(
        self,
        n_feat: int,
        hidden_dim: int,
        lr: float,
        output_dim: int  = 1,
        weight_decay: float = 0.):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay)

        self.rnn = nn.LSTM(input_size=n_feat, hidden_size=hidden_dim, batch_first=True)
        self.mlp = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self._bce = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs, _ = self.rnn(x)
        # use only the last step
        last_hs = hs[:, -1, :]
        return self.mlp(last_hs)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        (x, y) = batch
        yhat = self.forward(x)
        loss = self._bce(yhat, y)
        return loss


class LSTMRegressor(BaseLightningModule):
    def __init__(
        self,
        n_feat: int,
        hidden_dim: int,
        lr: float,
        output_dim: int = 1,
        weight_decay: float = 0.):
        super().__init__(lr=lr, weight_decay=weight_decay)

        self.rnn = nn.LSTM(input_size=n_feat, hidden_size=hidden_dim, batch_first=True)
        self.mlp = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self._mse = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs, _ = self.rnn(x)
        # use the last step
        last_hs = hs[:, -1, :]
        return self.mlp(last_hs)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        (x, y) = batch
        yhat = self.forward(x)
        loss = self._mse(yhat, y)
        return loss

# LAYERS FOR TEMPORAL CONVOLUTIONS

class CasualConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        (pad, ) = self.padding
        if pad > 0:
            seq_len = x.size(2)
            x = x[:, :, :seq_len - pad]
        return x.contiguous()    


class TemporalConvBlock(nn.Module):
    def __init__(self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        padding: int,
        n_layers: int = 2,
        ):
        
        super().__init__()
        
        channels = [input_dim] + [hidden_dim for _ in range(n_layers)]
        
        modules = []
        for in_channel, out_channel in zip(channels[: -1], channels[1: ]):
            conv = CasualConv1d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                )
            
            modules.append(conv)
            modules.append(nn.PReLU())
        
        self.model = nn.Sequential(*modules)
        self.up_or_downsample = (
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, dilation=1, padding=0)
            if input_dim != hidden_dim
            else None
        )


    def forward(self, x: torch.Tensor):
        skip_conn = x if self.up_or_downsample is None else self.up_or_downsample(x)
        # residual
        residual = self.model(x)
        # output to next layer
        return skip_conn + residual


class TranposeTimeSeries(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


