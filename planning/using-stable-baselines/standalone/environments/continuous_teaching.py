from abc import ABC

import gym
import numpy as np


class ContinuousTeaching(gym.Env, ABC):

    def __init__(self, alpha=1., beta=0.2, tau=0.9, n_item=30, t_max=1000):
        super().__init__()

        self.action_space = gym.spaces.Discrete(n_item)
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf,
                                                shape=(n_item, 2))
        self.state = np.zeros((n_item, 2))
        self.obs = np.zeros((n_item, 2))
        self.n_item = n_item
        self.t_max = t_max
        self.t = 0

        self.log_tau = np.log(tau)
        self.alpha = alpha
        self.beta = beta

    def reset(self):
        self.state = np.zeros((self.n_item, 2))
        self.obs = np.zeros((self.n_item, 2))
        self.t = 0
        return self.obs

    def step(self, action):
        self.state[:, 0] += 1  # delta + 1 for all
        self.state[action, 0] = 0  # ...except for item shown
        self.state[action, 1] += 1  # increment number of presentation

        done = self.t == self.t_max - 1

        view = self.state[:, 1] > 0
        delta = self.state[view, 0]
        rep = self.state[view, 1] - 1.

        forget_rate = self.alpha * (1 - self.beta) ** rep

        logp_recall = - forget_rate * delta
        above_thr = logp_recall > self.log_tau
        reward = np.count_nonzero(above_thr) / self.n_item

        self.obs[view, 0] = np.exp(-forget_rate * (delta + 1))
        self.obs[view, 1] = forget_rate

        info = {}
        self.t += 1
        return self.obs, reward, done, info
