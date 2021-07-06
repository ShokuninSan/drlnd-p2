from typing import Tuple, Union

import numpy as np


UnityAction = Union[int, np.ndarray]


class UnityEnvWrapper:
    """
    Wrapper for Unity environments exposing a Gym like API.
    """

    def __init__(self, unity_env):
        """
        Creates a Gym like API from a Unity environment.

        :param unity_env: an instance of a Unity environment.
        """
        self.env = unity_env
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.env_info.vector_observations.shape[1]

    def reset(self, train_mode: bool = True) -> np.ndarray:
        """
        Resets the environment.

        :param train_mode: toggles the training mode.
        :return: new state.
        """
        self.env_info = self.env.reset(train_mode)[self.brain_name]
        return self.env_info.vector_observations

    def step(self, action: UnityAction) -> Tuple[np.array, float, bool]:
        """
        Perform given action in the environment.

        :param action: action step.
        :return: (next_state, reward, done) tuple.
        """
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done

    def close(self) -> None:
        """
        Closes the environment.
        """
        self.env.close()
