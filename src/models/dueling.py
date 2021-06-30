from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDenseQNetwork(nn.Module):
    """
    A fully-connected Dueling Q-Network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int],
        activation_fn: Callable = F.relu,
        seed: Optional[int] = None,
    ):
        """
        Creates a fully-connected Dueling Q-Network instance.

        :param input_dim: dimension of input layer.
        :param output_dim: dimension of output layer.
        :param hidden_dims: dimensions of hidden layers.
        :param activation_fn: activation function (default: ReLU).
        :param seed: random seed.
        """
        super(DuelingDenseQNetwork, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.value_output = nn.Linear(hidden_dims[-1], 1)
        self.advantage_output = nn.Linear(hidden_dims[-1], output_dim)

        self.activation_fn = activation_fn

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass.

        :param state: state input.
        :return: action value output.
        """
        x = self.activation_fn(self.input_layer(state))
        for h in self.hidden_layers:
            x = self.activation_fn(h(x))

        a = self.advantage_output(x)
        v = self.value_output(x)
        v = v.expand_as(a)
        q = v + a - a.mean(1, keepdim=True).expand_as(a)

        return q
