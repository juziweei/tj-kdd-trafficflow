"""Temporal Fusion Transformer for traffic forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TFTConfig:
    hidden_size: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    patience: int = 10


class TemporalFusionTransformer:
    """Lightweight TFT implementation for time series forecasting."""

    def __init__(self, config: TFTConfig, input_dim: int, output_horizon: int):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TFT")

        self.config = config
        self.input_dim = input_dim
        self.output_horizon = output_horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scaler_mean = None
        self.scaler_std = None

    def _build_model(self):
        """Build simplified TFT architecture."""
        class SimpleTFT(nn.Module):
            def __init__(self, input_dim, hidden_size, num_heads, dropout, output_horizon):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, hidden_size)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=num_heads,
                        dim_feedforward=hidden_size * 4,
                        dropout=dropout,
                        batch_first=True
                    ),
                    num_layers=2
                )
                self.output_proj = nn.Linear(hidden_size, output_horizon)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                x = self.input_proj(x)
                x = self.dropout(x)
                x = self.encoder(x)
                x = x.mean(dim=1)
                return self.output_proj(x)

        model = SimpleTFT(
            self.input_dim,
            self.config.hidden_size,
            self.config.num_heads,
            self.config.dropout,
            self.output_horizon
        )
        return model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train TFT model."""
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self.scaler_mean) / self.scaler_std

        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = nn.functional.mse_loss(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained TFT."""
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_tensor)
        return pred.cpu().numpy()
