import torch.nn as nn

# Define the HNN for PDEs.

class PDHNN(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 hidden_layers):
        
        # Initialize the upper class.
        super(PDHNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        # Define the model.
        layers = [nn.Linear(input_dim, hidden_dim, bias=True),
                  nn.Tanh()]
        
        for i in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1, bias=True))

        # Model layers.
        self.model = nn.ModuleList(layers)

    def forward(self, x):

        for layer in self.model:
            x = layer(x)

        return x
    