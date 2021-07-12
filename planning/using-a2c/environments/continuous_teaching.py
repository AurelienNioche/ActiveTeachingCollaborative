from abc import ABC

import gym
import numpy as np


class ContinuousTeaching(gym.Env, ABC):

    def __init__(
            self,
            initial_forget_rates: np.ndarray,
            initial_repetition_rates: np.ndarray,
            tau=0.9,
            n_item=30,
            t_max=1000,
            time_per_iter=1,
            concat=False,
            interval=1
    ):
        super().__init__()

        self.action_space = gym.spaces.Discrete(n_item)
        self.state = np.zeros((n_item, 2))
        self.n_item = n_item
        self.t_max = t_max
        self.time_per_iter = time_per_iter
        self.log_tau = np.log(tau)

        if initial_repetition_rates.shape[0] == n_item and \
                initial_forget_rates.shape[0] == n_item:
            self.initial_forget_rates = initial_forget_rates
            self.initial_repetition_rates = initial_repetition_rates
        else:
            raise ValueError(
                "Mismatch between initial_rates shapes and n_item"
            )
        self.interval = interval
        self.obs_dim = 2
        self.obs = np.zeros((n_item, self.obs_dim))
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf,
                                                shape=(n_item * self.obs_dim, ))
        self.learned_before = np.zeros((self.n_item, ))
        self.t = 0

    def reset(self):
        self.state = np.zeros((self.n_item, 2))
        self.obs = np.zeros((self.n_item, self.obs_dim))
        self.obs[:, 1] = self.initial_forget_rates
        # self.learned_before = np.zeros((self.n_item, ))
        self.t = 0
        return self.obs.flatten()

    def step(self, action):

        self.state[:, 0] += self.time_per_iter  # add time elapsed since last iter
        self.state[action, 0] = 0               # ...except for item shown
        self.state[action, 1] += 1              # increment nb of presentation

        done = self.t == self.t_max - 1

        view = self.state[:, 1] > 0
        delta = self.state[view, 0]       # only consider already seen items
        rep = self.state[view, 1] - 1.    # only consider already seen items

        forget_rate = self.initial_forget_rates[view] * \
                      (1 - self.initial_repetition_rates[view]) ** rep

        logp_recall = - forget_rate * delta
        above_thr = logp_recall > self.log_tau
        # n_learned_now = np.count_nonzero(above_thr)
        #
        # penalizing_factor = n_learned_now - np.count_nonzero(self.learned_before)
        # penalizing_factor /= n_learned_now
        # print(n_learned_now)
        # print(penalizing_factor)

        reward = np.count_nonzero(above_thr) / self.n_item # + min(penalizing_factor, 0)

        self.learned_before = above_thr
        # Probability of recall at the time of the next action
        self.obs[view, 0] = np.exp(-forget_rate *
                                   (self.interval * delta + self.time_per_iter))
        # Forgetting rate of probability of recall
        self.obs[view, 1] = forget_rate

        info = {}
        self.t += 1
        return self.obs.flatten(), reward, done, info

    @classmethod
    def get_p_recall(cls, obs):
        obs = obs.reshape((obs.shape[0] // 2, 2))
        return obs[:, 0]
