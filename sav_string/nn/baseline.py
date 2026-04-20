import torch
import torch.nn as nn
from sav_string.nn.node import BaselineFlow


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



def load_baseline_model(
    architecture: str,
    input_dim: int,
    hyperparameters: dict
):

    # Load the model hyperparameters.
    if architecture != 'flow_map':
        model_hyperparams = hyperparameters[architecture]

    if architecture == 'mlp':

        model = MLPNextValue(
            input_dim=input_dim,
            hidden_sizes=model_hyperparams["hidden_sizes"]
        )

    elif architecture == 'cnn':

        model = ConvSequenceModel(
            input_dim=input_dim,
            hidden_dim=model_hyperparams["hidden_size"],
            output_dim=input_dim,
            n_layers=model_hyperparams["n_layers"],
            kernel_size=model_hyperparams["kernel_size"]
        )

    elif architecture == 'flow_map':

        model = BaselineFlow()

    return model