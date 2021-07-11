# -*- coding: utf-8 -*-
from copy import copy, deepcopy
import os
import numpy as np
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.optim as optim
from torch.nn import SmoothL1Loss

from models.actor_critic import DeterministicPolicyNetwork, FullyConnectedQNetwork
from experiences import ReplayBuffer, ExperienceBatch


ACTOR_CHECKPOINT_FILENAME = "drlnd_p2_actor.pth"
CRITIC_CHECKPOINT_FILENAME = "drlnd_p2_critic.pth"


class DDPG:
    """
    A deep deterministic policy-gradient agent.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        actor_hidden_layer_dimensions: Tuple[int] = (256, 128),
        critic_hidden_layer_dimensions: Tuple[int] = (256, 256, 128),
        buffer_size: int = 1000_000,
        batch_size: int = 128,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        seed: Optional[int] = None,
    ):
        """
        Creates an instance of a DDPG agent.

        :param state_size: size of state space.
        :param action_size: size of action space.
        :param actor_hidden_layer_dimensions: hidden layer dimensions of the policy network.
        :param critic_hidden_layer_dimensions: hidden layer dimensions of Q-network.
        :param buffer_size: replay buffer size.
        :param batch_size: mini-batch size.
        :param gamma: discount factor.
        :param tau: interpolation parameter for target-network weight update.
        :param lr_actor: learning rate of the policy network.
        :param lr_critic: learning rate of the Q-network.
        :param seed: random seed.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.critic_local = FullyConnectedQNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_dims=critic_hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)
        self.critic_target = FullyConnectedQNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_dims=critic_hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)

        self.actor_local = DeterministicPolicyNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_dims=actor_hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)
        self.actor_target = DeterministicPolicyNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_dims=actor_hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)

        self.value_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=self.lr_critic
        )
        self.policy_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.lr_actor
        )

        self.loss_fn = SmoothL1Loss()

        self.memory = ReplayBuffer(
            self.action_size, self.buffer_size, self.batch_size, self.seed
        )

        self.update_every = 4
        self.step_count = 0

    def _sample_noise(self) -> np.ndarray:
        """
        Samples noise from a Gaussian distribution.

        :return: ndarray with Gaussian distributed values of the same size as
        the action space.
        """
        return np.random.randn(self.action_size)

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
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1

        if (
            len(self.memory) > self.batch_size
            and (self.step_count % self.update_every) == 0
        ):
            experiences = self.memory.sample()
            self._optimize(experiences)

    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        """
        Returns actions for given state as per current policy.

        :param state: current state.
        :param eps: noise weighting coefficient.
        :return: selected action.
        """

        state = torch.from_numpy(state).float().to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        action += self._sample_noise() * eps

        return np.clip(action, -1, 1)

    def _optimize(
        self,
        experiences: ExperienceBatch,
    ) -> None:
        """
        Updates value parameters using given batch of experience tuples.

        :param experiences: tuple of (s, a, r, s', done) tuples.
        """
        states, actions, rewards, next_states, dones = experiences
        batch_size = len(dones)

        next_action = self.actor_target(next_states)
        next_q = self.critic_target(next_states, next_action).detach()

        target_q = rewards + self.gamma * next_q * (1 - dones)
        local_q = self.critic_local(states, actions)
        value_loss = self.loss_fn(local_q, target_q)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        actions = self.actor_local(states)
        policy_loss = -self.critic_local(states, actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._update_target_model(self.critic_local, self.critic_target)
        self._update_target_model(self.actor_local, self.actor_target)

    def _update_target_model(self, local_model, target_model) -> None:
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

    def fit(
        self,
        environment,
        n_episodes: int = 5000,
        max_t: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        scores_window_length: int = 100,
        average_target_score: float = 30.0,
        agent_checkpoint_dir: str = "",
    ) -> List[float]:
        """
        Trains the agent on the given environment.

        :param environment: environment instance to interact with.
        :param n_episodes: maximum number of training episodes.
        :param max_t:  maximum number of time steps per episode.
        :param eps_start: starting value of epsilon, controlling random noise in action selection.
        :param eps_end: minimum value of epsilon.
        :param eps_decay: multiplicative factor (per episode) for decreasing epsilon.
        :param scores_window_length: length of scores window to monitor convergence.
        :param average_target_score: average target score for scores_window_length at which learning stops.
        :param agent_checkpoint_dir: directory to store agent's model weights to.
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
                self.save(agent_checkpoint_dir)
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

    def save(self, agent_checkpoint_dir: str) -> None:
        """
        Stores the weights of the actor model.

        :param agent_checkpoint_dir: path to store agent's model weights to.
        """
        actor_checkpoint_path = os.path.join(agent_checkpoint_dir, ACTOR_CHECKPOINT_FILENAME)
        critic_checkpoint_path = os.path.join(agent_checkpoint_dir, CRITIC_CHECKPOINT_FILENAME)

        torch.save(self.actor_local.state_dict(), actor_checkpoint_path)
        torch.save(self.critic_local.state_dict(), critic_checkpoint_path)

    @staticmethod
    def load(agent_checkpoint_dir: str) -> "DDPG":
        """
        Loads the stored actor's weights into the local model and creates a DDPG agent instance.

        :param agent_checkpoint_dir: directory to load the actor model weights from.
        :return: a pre-trained agent instance.
        """
        actor_checkpoint_path = os.path.join(agent_checkpoint_dir, ACTOR_CHECKPOINT_FILENAME)
        state_dict = torch.load(actor_checkpoint_path)
        state_size = list(state_dict.values())[0].shape[1]
        action_size = list(state_dict.values())[-1].shape[0]
        agent = DDPG(state_size=state_size, action_size=action_size)
        agent.actor_local.load_state_dict(state_dict)
        return agent
