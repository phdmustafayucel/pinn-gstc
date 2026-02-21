import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Fully connected neural network used by PINNs.

    The network approximates a function:

        u_theta(x)

    where:
        x : input coordinates
        u : predicted physical quantity

    For the plane wave example:

        x -> E(x)

    Parameters
    ----------
    layers : list
        Example:
        [1, 64, 64, 1]

        meaning:
        input_dim = 1
        hidden layers = 64 neurons
        output_dim = 1
    """

    def __init__(self, layers):

        super().__init__()

        modules = []

        for i in range(len(layers) - 2):

            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.Tanh())

        modules.append(nn.Linear(layers[-2], layers[-1]))

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : tensor
            Input coordinates

        Returns
        -------
        tensor
            Predicted physical field
        """

        return self.network(x)