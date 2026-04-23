import torch.nn as nn


class DQNNetwork(nn.Module):
    '''
    This class defines a flexible feedforward network architecture.
    Maps state -> Q-values for each action.

    Parameters:
        input_size (int): Dimensionality of the input state space.
        hidden_layers (list[int]): List specifying the number of neurons in each hidden layer.
        output_size (int): Number of actions in the environment.
        activation_f (type[nn.Module]): Activation function to use between layers (optional - default: nn.ReLU).
        dropout_rate (float): Dropout rate to apply after each hidden layer (default: 0.0).
    '''

    def __init__(self,
                 input_size: int,
                 hidden_layers: list[int],
                 output_size: int,
                 activation_f: type[nn.Module] = nn.ReLU,
                 dropout_rate: float = 0.0):
        super().__init__()

        # Initialize empty list to store nn.Modules
        layers = []
        # Track output size of previous layer to use as input size for next layer
        prev_size = input_size

        # Build each hidden layer: append linear layer, activation, optional dropout
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(activation_f())

            if dropout_rate > 0:  # Condition to apply dropout
                layers.append(nn.Dropout(dropout_rate))

            prev_size = h

        # Output layer: 1 neuron for each action, no activation function
        layers.append(nn.Linear(prev_size, output_size))

        # Unpacks the list into a sequential pipeline
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)