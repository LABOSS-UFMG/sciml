# -------------------------------------------------------------------------------- #
import torch

from typing import Sequence

# -------------------------------------------------------------------------------- #
class MLP(torch.nn.Module):
    """
    Fully connected network with tanh hidden layers and a linear output.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: Sequence[int] = (32, 32),
            activation: torch.nn.Module = torch.nn.Tanh(),
        ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Number of input features.
        output_dim : int
            Number of output features.
        hidden_layers : Sequence[int], default=(32, 32)
            Width of each hidden layer, in order.
        activation : torch.nn.Module, default=torch.nn.Tanh()
            Activation function to use for the hidden layers.
        """
        super().__init__()

        # ------------------------------------------------------------------------ #
        # Build hidden layers and the linear output layer
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(previous_dim, hidden_dim))
            layers.append(activation)
            previous_dim = hidden_dim
        layers.append(torch.nn.Linear(previous_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -------------------------------------------------------------------------------- #
