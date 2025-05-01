import torch
import torch.nn as nn
from typing import Tuple, List


class TrendLayer(nn.Module):
    """
    Models polynomial trends in time series data.

    This layer learns polynomial coefficients for each feature dimension and applies them
    to generate trend patterns in the time series.

    Args:
        seq_len (int): Length of the time series sequence
        feat_dim (int): Number of features/dimensions in the time series
        latent_dim (int): Dimension of the latent space
        trend_poly (int): Degree of the polynomial trend to model
    """

    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int, trend_poly: int):
        super(TrendLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly

        self.trend_encoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly),
            nn.ReLU(),
            nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        trend_params: torch.Tensor = self.trend_encoder(z)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)

        lin_space: torch.Tensor = (
            torch.arange(0, float(self.seq_len), 1, device=z.device) / self.seq_len
        )
        poly_space: torch.Tensor = torch.stack(
            [lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0
        )

        trend: torch.Tensor = trend_params @ poly_space
        trend: torch.Tensor = trend.transpose(2, 1)
        return trend


class SeasonalLayer(nn.Module):
    """
    Models seasonal patterns in time series data.

    This layer can handle multiple seasonal patterns with different periods and lengths.
    Each seasonal pattern is learned independently and combined additively.

    Args:
        seq_len (int): Length of the time series sequence
        feat_dim (int): Number of features/dimensions in the time series
        latent_dim (int): Dimension of the latent space
        custom_seas (List[Tuple[int, int]]): List of (num_seasons, len_per_season) tuples
            specifying the seasonal patterns to model
    """

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        latent_dim: int,
        custom_seas: List[Tuple[int, int]],
    ):
        super(SeasonalLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.custom_seas = custom_seas

        self.input_proj_layers = nn.ModuleList(
            [
                nn.Linear(latent_dim, feat_dim * num_seasons)
                for num_seasons, _ in custom_seas
            ]
        )

    def _get_season_indexes_over_seq(self, num_seasons: int, len_per_season: int):
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.view(-1)
        season_indexes = season_indexes.repeat(self.seq_len // len_per_season + 1)[
            : self.seq_len
        ]
        return season_indexes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        N = z.size(0)
        ones_tensor = torch.ones(
            (N, self.feat_dim, self.seq_len), dtype=torch.int32, device=z.device
        )

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params: torch.Tensor = self.input_proj_layers[i](z)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)

            season_indexes_over_time: torch.Tensor = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            ).to(z.device)

            dim2_idxes: torch.Tensor = ones_tensor * season_indexes_over_time.view(
                1, 1, -1
            )
            season_vals: torch.Tensor = torch.gather(season_params, 2, dim2_idxes)
            all_seas_vals.append(season_vals)

        all_seas_vals: torch.Tensor = torch.stack(all_seas_vals, dim=-1)
        all_seas_vals: torch.Tensor = torch.sum(all_seas_vals, dim=-1)
        all_seas_vals: torch.Tensor = all_seas_vals.transpose(2, 1)
        return all_seas_vals


class LevelModel(nn.Module):
    """
    Models the base level of the time series.

    This layer learns a constant level for each feature dimension that persists
    throughout the entire sequence.

    Args:
        latent_dim (int): Dimension of the latent space
        feat_dim (int): Number of features/dimensions in the time series
        seq_len (int): Length of the time series sequence
    """

    def __init__(self, latent_dim: int, feat_dim: int, seq_len: int):
        super(LevelModel, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.level_encoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # (bs, feat_dim)
        level_params: torch.Tensor = self.level_encoder(z)

        level_params: torch.Tensor = level_params.view(-1, 1, self.feat_dim)
        ones_tensor = torch.ones(
            (1, self.seq_len, 1), dtype=torch.float32, device=z.device
        )

        level_vals: torch.Tensor = level_params * ones_tensor
        return level_vals


class ResidualConnection(nn.Module):
    """
    Models non-linear patterns and noise in the time series.

    This layer uses a convolutional decoder architecture to capture complex,
    non-linear patterns that cannot be explained by trend or seasonality.

    Args:
        seq_len (int): Length of the time series sequence
        feat_dim (int): Number of features/dimensions in the time series
        hidden_layer_sizes (List[int]): List of hidden layer sizes for the decoder
        latent_dim (int): Dimension of the latent space
        encoder_last_dense_dim (int): Dimension of the encoder's last dense layer
    """

    def __init__(
        self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim
    ):
        super(ResidualConnection, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.input_proj = nn.Linear(latent_dim, encoder_last_dense_dim)

        self.input_channels = hidden_layer_sizes[-1]
        self.input_dim = encoder_last_dense_dim // self.input_channels

        decoder_channels = [*hidden_layer_sizes[::-1], feat_dim]

        n_layers = len(hidden_layer_sizes)
        output_sequence_len = self.input_dim * (2**n_layers)
        final = nn.Linear(output_sequence_len * feat_dim, seq_len * feat_dim)

        decoder_layers = []
        for i in range(len(decoder_channels) - 1):
            convt = nn.ConvTranspose1d(
                decoder_channels[i],
                decoder_channels[i + 1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
            decoder_layers.append(convt)
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers, nn.Flatten(), final)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.input_proj(z)
        x: torch.Tensor = x.view(-1, self.input_channels, self.input_dim)
        x: torch.Tensor = self.decoder(x)
        return x.view(-1, self.seq_len, self.feat_dim)


class TimeVAEEncoder(nn.Module):
    """
    Encodes time series data into a latent space representation.

    Uses a convolutional architecture to compress the time series into a lower-dimensional
    latent space, outputting both mean and log variance of the latent distribution.

    Args:
        seq_len (int): Length of the time series sequence
        feat_dim (int): Number of features/dimensions in the time series
        hidden_layer_sizes (List[int]): List of hidden layer sizes for the encoder
        latent_dim (int): Dimension of the latent space
    """

    def __init__(
        self, seq_len: int, feat_dim: int, hidden_layer_sizes: int, latent_dim: int
    ):
        super(TimeVAEEncoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        encoder_layers = []

        filter_dims = [feat_dim, *hidden_layer_sizes]

        for i in range(len(filter_dims) - 1):
            conv_layer = nn.Conv1d(
                filter_dims[i], filter_dims[i + 1], kernel_size=3, stride=2, padding=1
            )
            encoder_layers.append(conv_layer)
            encoder_layers.append(nn.ReLU())

        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_last_dense_dim = self._get_last_dense_dim(seq_len, feat_dim)
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = x.transpose(1, 2)
        x: torch.Tensor = self.encoder(x)
        z_mean: torch.Tensor = self.z_mean(x)
        z_log_var: torch.Tensor = self.z_log_var(x)
        return z_mean, z_log_var

    def _get_last_dense_dim(self, seq_len: int, feat_dim: int) -> int:
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            x: torch.Tensor = self.encoder(x)
            return x.numel()


class TimeVAEDecoder(nn.Module):
    """
    Decodes latent space representations back into time series data.

    Combines multiple components (trend, seasonality, level, and residuals) to reconstruct
    the original time series from its latent representation.

    Args:
        seq_len (int): Length of the time series sequence
        feat_dim (int): Number of features/dimensions in the time series
        hidden_layer_sizes (List[int]): List of hidden layer sizes for the decoder
        latent_dim (int): Dimension of the latent space
        encoder_last_dense_dim (int): Dimension of the encoder's last dense layer
        trend_poly (int, optional): Degree of polynomial trend to model. Defaults to 0
        custom_seas (List[Tuple[int, int]], optional): List of (num_seasons, len_per_season)
            tuples specifying seasonal patterns. Defaults to None
        use_residual_conn (bool, optional): Whether to use residual connections.
            Defaults to True
    """

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        hidden_layer_sizes: List[int],
        latent_dim: int,
        encoder_last_dense_dim: int,
        trend_poly: int = 0,
        custom_seas: List[Tuple[int, int]] = None,
        use_residual_conn: bool = True,
    ):
        super(TimeVAEDecoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder_last_dense_dim = encoder_last_dense_dim
        self.level_model = LevelModel(self.latent_dim, self.feat_dim, self.seq_len)

        if use_residual_conn:
            self.residual_conn = ResidualConnection(
                seq_len,
                feat_dim,
                hidden_layer_sizes,
                latent_dim,
                encoder_last_dense_dim,
            )

        if self.trend_poly is not None and self.trend_poly > 0:
            self.trend_layer = TrendLayer(
                self.seq_len, self.feat_dim, self.latent_dim, self.trend_poly
            )

        if self.custom_seas is not None and len(self.custom_seas) > 0:
            self.seasonal_layer = SeasonalLayer(
                self.seq_len, self.feat_dim, self.latent_dim, self.custom_seas
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        outputs: torch.Tensor = self.level_model(z)
        if self.trend_poly is not None and self.trend_poly > 0:
            trend: torch.Tensor = self.trend_layer(z)
            outputs += trend

        if self.custom_seas is not None and len(self.custom_seas) > 0:
            seasonality: torch.Tensor = self.seasonal_layer(z)
            outputs += seasonality

        if self.use_residual_conn:
            residuals: torch.Tensor = self.residual_conn(z)
            outputs += residuals

        return outputs
