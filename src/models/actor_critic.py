# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Callable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return -lim, lim


class DeterministicPolicyNetwork(nn.Module):
    """
    A fully-connected Q-value network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int],
        activation_fn: Callable = F.relu,
        output_activation_fn: Callable = F.tanh,
        seed: Optional[int] = None,
    ):
        """
        Creates a deterministic policy network instance.

        :param input_dim: dimension of input layer.
        :param output_dim: dimension of output layer.
        :param hidden_dims: dimensions of hidden layers.
        :param activation_fn: activation function (default: ReLU).
        :param output_activation_fn: output activation function (default: tanh).
        :param seed: random seed.
        """
        super(DeterministicPolicyNetwork, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i == 0:
                self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn

        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for h in self.hidden_layers:
            h.weight.data.uniform_(*hidden_init(h))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass.

        :param state: state input.
        :return: action value output.
        """
        x = self.activation_fn(self.input_layer(state))

        for h in self.hidden_layers:
            x = self.activation_fn(h(x))

        x = self.output_activation_fn(self.output_layer(x))
        return x


class FullyConnectedQNetwork(nn.Module):
    """
    A fully-connected Q-value network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int],
        activation_fn: Callable = F.leaky_relu,
        seed: Optional[int] = None,
    ):
        """
        Creates a fully-connected Q-value network instance.

        :param input_dim: dimension of input layer.
        :param output_dim: dimension of output layer.
        :param hidden_dims: dimensions of hidden layers.
        :param activation_fn: activation function (default: ReLU).
        :param seed: random seed.
        """
        super(FullyConnectedQNetwork, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            _input_dim = hidden_dims[i]
            if i == 0:
                _input_dim += output_dim
                self.hidden_layers.append(nn.Linear(_input_dim, hidden_dims[i + 1]))
                self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            else:
                self.hidden_layers.append(nn.Linear(_input_dim, hidden_dims[i + 1]))

        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.activation_fn = activation_fn

        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for h in self.hidden_layers:
            h.weight.data.uniform_(*hidden_init(h))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass.

        :param state: state input.
        :return: action value output.
        """
        x = self.activation_fn(self.input_layer(state))

        for i, h in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat([x, action], dim=1)
            x = self.activation_fn(h(x))

        return self.output_layer(x)
