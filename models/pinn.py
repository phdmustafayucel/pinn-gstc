import torch.nn as nn
from models.mlp import MLP


class PINN(nn.Module):
    """
    Generic PINN model.

    This class wraps the neural network that approximates the
    physical solution.

    The neural network represents:

        u_theta(x)

    where theta are trainable parameters.
    """

    def __init__(self, layers):

        super().__init__()

        self.model = MLP(layers)

    def forward(self, x):
        """
        Evaluate the neural network.

        Parameters
        ----------
        x : tensor

        Returns
        -------
        tensor
            Predicted solution
        """

        return self.model(x)