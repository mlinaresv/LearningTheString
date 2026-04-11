import torch
import torch.nn as nn
import math


# Positional Encoding (standard sine/cosine)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class TransformerStateEstimator(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        output_dim=1,
        dropout=0.1
    ):
        super().__init__()

        self.hidden_state_dim = output_dim // 2
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LeakyReLU(),
            nn.Linear(model_dim, output_dim)
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """

        # Reshape the input of the transformer.
        x = x.unsqueeze(-1)

        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Use last timestep representation
        last_token = x[:, -1, :]  # (batch_size, model_dim)
        
        # Predict state
        out = self.head(last_token)
        
        return out[..., :self.hidden_state_dim], out[..., self.hidden_state_dim:]