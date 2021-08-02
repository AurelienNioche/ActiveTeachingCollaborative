import random
from abc import ABC

import gym
import numpy as np

types = {
    'monotonic': 1,
    'exam_based': 2,
    'mean_learned': 3
}

class ContinuousTeaching(gym.Env, ABC):

    def __init__(
            self,
            initial_forget_rates: np.ndarray,
            initial_repetition_rates: np.ndarray,
            delta_coeffs: np.array,
            tau=0.9,
            n_item=30,
            t_max=1000,
            time_per_iter=1,
            n_coeffs: int = 1,
            penalty_coeff: float = 0.2,
            reward_coeff: float = 1,
            reward_type=types['monotonic']
    ):
        super().__init__()

        self.action_space = gym.spaces.Discrete(n_item)
        self.state = np.zeros((n_item, 2))
        self.n_item = n_item
        self.t_max = t_max
        self.time_per_iter = time_per_iter
        self.tau = tau
        self.log_tau = np.log(self.tau)

        if delta_coeffs.shape[0] != n_coeffs:
            raise ValueError(
                "Mismatch between delta_coeffs shape and n_coeffs"
            )
        self.delta_coeffs = delta_coeffs

        self.obs_dim = n_coeffs + 1
        self.obs = np.zeros((n_item, self.obs_dim))
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf,
                                                shape=(n_item * self.obs_dim, ))
        self.learned_before = np.zeros((self.n_item, ))
        self.t = 0
        self.penalty_coeff = penalty_coeff
        random.seed(123)
        self.n_users = initial_forget_rates.shape[0]
        self.current_user = self.pick_a_user()
        if initial_repetition_rates.shape[1] == n_item and \
                initial_forget_rates.shape[1] == n_item:
            self.all_forget_rates = initial_forget_rates
            self.all_repetition_rates = initial_repetition_rates
            self.initial_forget_rates = initial_forget_rates[self.current_user]
            self.initial_repetition_rates = initial_repetition_rates[self.current_user]
        else:
            raise ValueError(
                "Mismatch between initial_rates shapes and n_item"
            )
        # self.reward_range = (- reward_coeff, reward_coeff)
        self.reward_coeff = reward_coeff
        self.reward_type = reward_type

    def pick_a_user(self):
        self.current_user = random.randint(0, self.n_users - 1)
        return self.current_user

    def reset(self, user=None):
        if not user:
            user = self.pick_a_user()
        self.current_user = user
        self.state = np.zeros((self.n_item, 2))
        self.initial_forget_rates = self.all_forget_rates[user]
        self.initial_repetition_rates = self.all_repetition_rates[user]
        self.obs = np.zeros((self.n_item, self.obs_dim))
        self.obs[:, 2] = self.initial_repetition_rates
        self.learned_before = np.zeros((self.n_item, ))
        self.t = 0
        return self.obs.flatten()

    def compute_reward(self, forget_rate, delta):

        logp_recall = - forget_rate * delta
        above_thr = logp_recall > self.log_tau
        n_learned_now = np.count_nonzero(above_thr)

        if self.reward_type == types['monotonic']:
            learned_diff = n_learned_now - np.count_nonzero(self.learned_before)
            reward = learned_diff

        elif self.reward_type == types['mean_learned']:
            penalizing_factor = n_learned_now - np.count_nonzero(self.learned_before)
            # penalizing_factor /= n_learned_now

            reward = (1 - self.penalty_coeff) * (np.count_nonzero(above_thr) / self.n_item) \
                     + self.penalty_coeff * penalizing_factor

        elif self.reward_type == types['exam_based']:
            if self.t == self.t_max - 1:
                reward = n_learned_now / self.n_item
            else:
                reward = 0

        reward *= self.reward_coeff
        self.learned_before = above_thr
        return reward

    def step(self, action):

        self.state[:, 0] += self.time_per_iter  # add time elapsed since last iter
        self.state[action, 0] = 0               # ...except for item shown
        self.state[action, 1] += 1              # increment nb of presentation

        done = self.t == self.t_max - 1

        view = self.state[:, 1] > 0
        delta = self.state[view, 0]  # only consider already seen items
        rep = self.state[view, 1] - 1.  # only consider already seen items

        forget_rate = self.initial_forget_rates[view] * \
                      (1 - self.initial_repetition_rates[view]) ** rep

        reward = self.compute_reward(forget_rate, delta)

        # Probability of recall at the time of the next action
        for i in range(self.delta_coeffs.shape[0]):
            self.obs[view, i] = np.exp(
                -forget_rate *
                (self.delta_coeffs[i] * delta + self.time_per_iter)
            )

        info = {}
        self.t += 1
        return self.obs.flatten(), reward, done, info

    @classmethod
    def get_p_recall(cls, obs):
        obs = obs.reshape((obs.shape[0] // 2, 2))
        return obs[:, 0]
