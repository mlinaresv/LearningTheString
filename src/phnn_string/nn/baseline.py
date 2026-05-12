import math
import torch
import torch.nn as nn


class MLPNextValue(nn.Module):

    def __init__(
        
        self, 
        input_dim, 
        hidden_sizes=[128, 256, 512]
    
    ):

        super(MLPNextValue, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        # Hidden layers
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.LeakyReLU())
            prev_size = h
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_size, input_dim))
        
        self.model = nn.Sequential(*layers)
    

    def forward(self, x, n_steps):

        outputs = []

        current_input = x
        for _ in range(n_steps):
            out = self.model(current_input)  # (batch_size, output_dim)
            outputs.append(out.unsqueeze(1))  # (batch_size, 1, output_dim)

            # Optional: feed output back as next input (depends on your use case)
            current_input = out

        return torch.cat(outputs, dim=1)  # (batch_size, n_steps, output_dim)


    def rollout(self, x_init, steps, mean_pos=None, std_pos=None, device='cpu'):
        """
        Recursive prediction over multiple steps.

        Args:
            x_init: (batch_size, N) initial sequence
            steps: number of future steps to predict

        Returns:
            preds: (batch_size, steps)
        """
        preds = []
        x = x_init.clone()

        # Add the initial state.
        preds.append(x[:, -1:].detach().cpu())

        for _ in range(steps):

            if mean_pos is not None and std_pos is not None:
                x_in = (x - mean_pos) / std_pos
            else:
                x_in = x

            y_out = self.forward(x_in.to(device), n_steps=1).squeeze(1).detach().cpu()            # (batch, 1)

            if mean_pos is not None and std_pos is not None:
                y = (y_out * std_pos) + mean_pos
            else:
                y = y_out

            preds.append(y[:, -1:])

            # shift window: drop first value, append prediction
            x = torch.cat([x[:, 1:].detach().cpu(), y[:, -1:]], dim=1)

        return torch.cat(preds, dim=1)



class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding]  # remove future leakage


class ConvSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=4, kernel_size=3):

        super().__init__()

        layers = []
        for i in range(n_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(CausalConv1d(in_ch, hidden_dim, kernel_size, dilation=2**i))
            layers.append(nn.LeakyReLU())

        self.net = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)


    def forward(self, x, n_steps):

        # x: (B, input_dim)
        x = x.unsqueeze(1)
        x = x.transpose(1, 2)  # (B, input_dim, T)

        outputs = []

        current_input = x 
        for _ in range(n_steps):
            h = self.net(current_input)
            out = self.head(h)

            outputs.append(out.transpose(1, 2))
            current_input = out

        return torch.cat(outputs, dim=1)


    @torch.no_grad()
    def rollout(self, x_init, n_steps, mean_pos=None, std_pos=None, device='cpu'):
        preds = []
        x = x_init.clone()

        # Add the initial state.
        preds.append(x[:, -1:].detach().cpu())

        for _ in range(n_steps):

            if mean_pos is not None and std_pos is not None:
                x_in = (x - mean_pos) / std_pos
            else:
                x_in = x

            y_out = self.forward(x_in.to(device), n_steps=1).squeeze(1).detach().cpu()

            if mean_pos is not None and std_pos is not None:
                y = (y_out * std_pos) + mean_pos
            else:
                y = y_out

            preds.append(y[:, -1:])

            x = torch.cat([x[:, 1:].detach().cpu(), y[:, -1:]], dim=1)

        return torch.cat(preds, dim=1)



class BaselineFlow(nn.Module):
    def __init__(
            
            self, 
            input_dim=406, 
            hidden1=64, 
            hidden2=32, 
            output_dim=404
            
        ):

        super().__init__()

        self.n_states = output_dim // 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LeakyReLU(),

            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(),

            nn.Linear(hidden2, output_dim)
        )


    def forward(self, q, p, u, exc_pos):
        """
        x: (batch_size, 406)
        returns: (batch_size, 404) -> next state (p_next, q_next flattened)
        """

        x = torch.cat((q, p, u, exc_pos), dim=-1)
        next_x = self.net(x)

        return next_x[..., :self.n_states], next_x[..., self.n_states:]
    

    def rollout(self, q0, p0, u, exc_pos, lib):
        
        time_steps = u.shape[-1] - 1
        p_trajectory, q_trajectory = lib.zeros((*p0.shape[:-1], time_steps + 1, p0.shape[-1])), lib.zeros((*q0.shape[:-1], time_steps + 1, q0.shape[-1]))

        # State initialization.
        qlast, plast = q0, p0
        p_trajectory[..., 0, :] = plast.detach().cpu()
        q_trajectory[..., 0, :] = qlast.detach().cpu()

        # Main loop.
        with torch.no_grad():
            for i in range(time_steps):

                u_t = u[:, i].unsqueeze(-1)
                qnext, pnext = self.forward(qlast, plast, u_t, exc_pos.to(qlast.device))
                qlast = qnext
                plast = pnext

                # Append the new generated states to the final trajectory.
                p_trajectory[..., i + 1, :] = pnext.detach().cpu()
                q_trajectory[..., i + 1, :] = qnext.detach().cpu()

        return p_trajectory, q_trajectory
    


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

