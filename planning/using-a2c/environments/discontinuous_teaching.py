from typing import Union

import gym
import numpy as np


class DiscontinuousTeaching(gym.Env):

    def __init__(
            self,                              # Setting previous XP
            initial_forget_rates: np.ndarray,
            initial_repetition_rates: np.ndarray,
            tau: float = 0.9,
            n_item: int = 30,                       # 500
            n_session: int = 6,                     # 6
            n_iter_per_session: int = 100,          # 100
            break_length: Union[float, int] = 10,   # 24*60**2
            time_per_iter: Union[float, int] = 1):  # 4

        super().__init__()

        self.action_space = gym.spaces.Discrete(n_item)
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf,
                                                shape=(n_item * 2 + 1,))
        self.n_item = n_item

        self.log_tau = np.log(tau)
        if initial_repetition_rates.shape[0] == n_item and \
                initial_forget_rates.shape[0] == n_item:
            self.initial_forget_rates = initial_forget_rates
            self.initial_repetition_rates = initial_repetition_rates
        else:
            raise ValueError(
                "Mismatch between initial_rates shapes and n_item"
            )

        self.n_session = n_session
        self.n_iter_per_session = n_iter_per_session
        self.break_length = break_length
        self.time_per_iter = time_per_iter

        # Things that need to be reset
        self.state = np.zeros((n_item, 2))
        self.obs = np.zeros((n_item, 2))
        self.current_iter = 0
        self.current_ss = 0
        self.time_elapsed_since_last_iter = 0

    def reset(self):
        self.state = np.zeros((self.n_item, 2))
        self.obs = np.zeros((self.n_item, 2))
        self.obs[:, 1] = self.initial_forget_rates
        self.current_iter = 0
        self.current_ss = 0
        self.time_elapsed_since_last_iter = 0
        return self.format_obs(0.)

    def format_obs(self, session_progression):
        return np.hstack(
            (self.obs.flatten(), np.array([session_progression, ])))

    def step(self, action):

        # increase delta for all items
        self.state[:, 0] += self.time_elapsed_since_last_iter
        # ...except for item shown
        self.state[action, 0] = 0
        # increment number of presentation
        self.state[action, 1] += 1

        view = self.state[:, 1] > 0
        delta = self.state[view, 0]
        rep = self.state[view, 1] - 1.

        forget_rate = self.initial_forget_rates[view] * \
              (1 - self.initial_repetition_rates[view]) ** rep
        logp_recall = - forget_rate * delta
        above_thr = logp_recall > self.log_tau
        reward = np.count_nonzero(above_thr) / self.n_item

        time_before_next_iter, done = self.next_delta()
        # Probability of recall at the time of the next action
        self.obs[view, 0] = np.exp(-forget_rate * (delta + time_before_next_iter))
        # Forgetting rate of probability of recall
        self.obs[view, 1] = forget_rate

        # update for next call
        self.time_elapsed_since_last_iter = time_before_next_iter

        # Get session progression at the time of the next action
        session_progression = self.session_progression()

        info = {}
        return self.format_obs(session_progression), reward, done, info

    @classmethod
    def extract_p_recall(cls, obs):
        obs = obs[:-1].reshape(((obs.shape[0] - 1) // 2, 2))
        return obs[:, 0]

    def next_delta(self):

        done = False

        self.current_iter += 1
        if self.current_iter >= self.n_iter_per_session:
            self.current_iter = 0
            self.current_ss += 1
            delta = self.break_length
        else:
            delta = self.time_per_iter

        if self.current_ss >= self.n_session:
            done = True

        return delta, done

    def session_progression(self):
        progress = self.current_iter / (self.n_iter_per_session - 1)
        return progress
