from torch import nn

import torch.nn as nn
import torch.nn.init as init

class BiGRUBaseline(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.linear = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x, **kwargs):
        # x: (B, T, F)
        out, _ = self.gru(x)
        out = self.linear(out)
        return out

class TransformerBaseline(nn.Module):
    def __init__(self, input_size, d_model=64, num_layers=2, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_size)

    def forward(self, x, **kwargs):
        # x: (B, T, F)
        x_proj = self.input_proj(x)
        encoded = self.encoder(x_proj)
        out = self.output_proj(encoded)
        return out
