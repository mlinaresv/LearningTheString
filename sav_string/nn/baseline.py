import torch
import torch.nn as nn

class MLPNextValue(nn.Module):

    def __init__(
        
        self, 
        input_size, 
        hidden_sizes=[128, 256, 512]
    
    ):

        super(MLPNextValue, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.LeakyReLU())
            prev_size = h
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_size, input_size))
        
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


    def rollout(self, x_init, steps):
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
        preds.append(x[:, -1:])

        for _ in range(steps):
            y = self.forward(x, n_steps=1).squeeze(1)            # (batch, 1)
            preds.append(y[:, -1:])

            # shift window: drop first value, append prediction
            x = torch.cat([x[:, 1:], y[:, -1:]], dim=1)

        return torch.cat(preds, dim=1).detach().cpu()