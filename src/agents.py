import numpy as np
import random
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.optim as optim
from torch.nn import SmoothL1Loss

from models.dueling import DuelingDenseQNetwork
from experiences import ReplayBuffer, ExperienceBatch


class DuelingDDQN:
    """
    A dueling double-DQN agent.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layer_dimensions: Tuple[int] = (512, 256),
        buffer_size: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr: float = 5e-4,
        update_every: int = 4,
        seed: Optional[int] = None,
    ):
        """
        Creates an instance of DuelingDDQN.

        :param state_size: size of state space.
        :param action_size: size of action space.
        :param hidden_layer_dimensions: dimensions of Q-network layer dims.
        :param buffer_size: replay buffer size.
        :param batch_size: mini-batch size.
        :param gamma: discount factor.
        :param tau: interpolation parameter for target-network weight update.
        :param lr: learning rate.
        :param update_every: how often to update the network.
        :param seed: random seed.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = DuelingDenseQNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_dims=hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)
        self.qnetwork_target = DuelingDenseQNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_dims=hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.loss_fn = SmoothL1Loss()
        self.memory = ReplayBuffer(
            self.action_size, self.buffer_size, self.batch_size, self.seed
        )

        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

    def _step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Adds the experience to memory and fits the agent.

        :param state: state of the environment.
        :param action: action taken.
        :param reward: reward received.
        :param next_state: next state after taken action.
        :param done: indicated if the episode has finished.
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, sample subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self._fit(experiences)

    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        """
        Returns actions for given state as per current policy.

        :param state: current state.
        :param eps: epsilon, for epsilon-greedy action selection.
        :return: selected action.
        """
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def _fit(
        self,
        experiences: ExperienceBatch,
    ) -> None:
        """
        Updates value parameters using given batch of experience tuples.

        :param experiences: tuple of (s, a, r, s', done) tuples.
        """
        states, actions, rewards, next_states, dones = experiences
        batch_size = len(dones)

        # get argmax (action) of online model
        next_argmax_q = self.qnetwork_local(next_states).max(1)[1]
        # get action values from target network
        next_q = self.qnetwork_target(next_states).detach()
        next_max_q = next_q[np.arange(batch_size), next_argmax_q].unsqueeze(1)

        next_max_q *= 1 - dones
        target_q = rewards + self.gamma * next_max_q
        local_q = self.qnetwork_local(states).gather(1, actions)
        loss = self.loss_fn(local_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_target_network(self.qnetwork_local, self.qnetwork_target)

    def _update_target_network(self, local_model, target_model) -> None:
        """
        Updates model parameters of target network using Polyak Averaging:

            θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: weights will be copied from.
        :param target_model: weights will be copied to.
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_weight_ratio = (1.0 - self.tau) * target_param.data
            local_weight_ratio = self.tau * local_param.data
            target_param.data.copy_(target_weight_ratio + local_weight_ratio)

    def learn(
        self,
        environment,
        n_episodes: int = 2000,
        max_t: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        scores_window_length: int = 100,
        average_target_score: float = 13.0,
        model_checkpoint_path: str = "checkpoint.pth",
    ) -> List[float]:
        """
        Trains the agent on the given environment.

        :param environment: environment instance to interact with.
        :param n_episodes: maximum number of training episodes.
        :param max_t:  maximum number of time steps per episode.
        :param eps_start: starting value of epsilon, for epsilon-greedy action selection.
        :param eps_end: minimum value of epsilon.
        :param eps_decay: multiplicative factor (per episode) for decreasing epsilon.
        :param scores_window_length: length of scores window to monitor convergence.
        :param average_target_score: average target score for scores_window_length at which learning stops.
        :param model_checkpoint_path: path to store model weights to.
        :return: list of scores.
        """
        scores = []
        scores_window = deque(maxlen=scores_window_length)
        eps = eps_start
        for i_episode in range(1, n_episodes + 1):
            state = environment.reset(train_mode=True)
            score = 0
            for t in range(max_t):
                action = self.act(state, eps)
                next_state, reward, done = environment.step(action)
                self._step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay * eps)
            average_score_window = float(np.mean(scores_window))
            self._log_progress(i_episode, average_score_window, scores_window_length)
            if np.mean(scores_window) >= average_target_score:
                print(
                    f"\nEnvironment solved in {i_episode:d} episodes!\t"
                    f"Average Score: {average_score_window:.2f}"
                )
                torch.save(self.qnetwork_local.state_dict(), model_checkpoint_path)
                break
        return scores

    @staticmethod
    def _log_progress(
        i_episode: int, average_score_window: float, scores_window_length: int
    ) -> None:
        """
        Logs average score of episode to stdout.

        :param i_episode: number of current episode.
        :param average_score_window: average score of current episode.
        :param scores_window_length: length of window for computing the average.
        """
        print(
            f"\rEpisode {i_episode}\tAverage Score: {average_score_window:.2f}",
            end="\n" if i_episode % scores_window_length == 0 else "",
        )

    @staticmethod
    def load(model_checkpoint_path: str) -> "DuelingDDQN":
        """
        Creates an agent and loads stored weights into the local model.

        :param model_checkpoint_path: path to load model weights from.
        :return: a pre-trained agent instance.
        """
        state_dict = torch.load(model_checkpoint_path)
        state_size = list(state_dict.values())[0].shape[1]
        action_size = list(state_dict.values())[-1].shape[0]
        agent = DuelingDDQN(state_size=state_size, action_size=action_size)
        agent.qnetwork_local.load_state_dict(state_dict)
        return agent


class DDPG:
    """
    A Deep Deterministic Policy Gradient agent.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layer_dimensions: Tuple[int] = (512, 256),
        buffer_size: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr: float = 5e-4,
        update_every: int = 4,
        seed: Optional[int] = None,
    ):
        """
        Creates an instance of DuelingDDQN.

        :param state_size: size of state space.
        :param action_size: size of action space.
        :param hidden_layer_dimensions: dimensions of Q-network layer dims.
        :param buffer_size: replay buffer size.
        :param batch_size: mini-batch size.
        :param gamma: discount factor.
        :param tau: interpolation parameter for target-network weight update.
        :param lr: learning rate.
        :param update_every: how often to update the network.
        :param seed: random seed.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = DuelingDenseQNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_dims=hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)
        self.qnetwork_target = DuelingDenseQNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_dims=hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.loss_fn = SmoothL1Loss()
        self.memory = ReplayBuffer(
            self.action_size, self.buffer_size, self.batch_size, self.seed
        )

        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0