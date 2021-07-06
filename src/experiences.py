import random
from collections import deque
from typing import Tuple, Optional
from dataclasses import dataclass

import torch
import numpy as np


ExperienceBatch = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


@dataclass
class Experience:
    """
    Experience data of an agent.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(
        self,
        action_size: int,
        buffer_size: int,
        batch_size: int,
        seed: Optional[int] = None,
    ):
        """
        Creates a ReplayBuffer instance.

        :param action_size: dimension of each action.
        :param buffer_size: maximum size of buffer.
        :param batch_size: size of each training batch.
        :param seed: random seed.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        if seed is not None:
            random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Adds new experience to the internal memory.

        :param state: current state of the environment.
        :param action: action taken.
        :param reward: reward received for given action.
        :param next_state: next state after taken the given action.
        :param done: indicates if episode has finished.
        """
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(
        self,
    ) -> ExperienceBatch:
        """
        Randomly sample a batch of experiences from memory.

        :return: batch of experiences.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Return the current size of internal memory.

        :return: size of internal memory.
        """
        return len(self.memory)
