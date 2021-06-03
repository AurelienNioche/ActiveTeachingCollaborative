import numpy as np


class Threshold:
    def __init__(self, env, tau):
        self.tau = tau
        self.n_item = env.n_item

    @staticmethod
    def extract_p_rec(obs):

        # If environment is ContinuousTeaching
        if obs.shape[0] % 2 == 0:
            obs = obs.reshape((obs.shape[0] // 2, 2))
        # If environment is DiscontinuousTeaching
        else:
            obs = obs[:-1].reshape(((obs.shape[0] - 1) // 2, 2))
        return obs[:, 0]

    def act(self, obs):

        p_rec = self.extract_p_rec(obs)

        view_under_thr = (0 < p_rec) * (p_rec <= self.tau)
        if np.count_nonzero(view_under_thr) > 0:
            items = np.arange(self.n_item)
            selection = items[view_under_thr]
            action = selection[np.argmin(p_rec[view_under_thr])]
        else:
            n_seen = np.count_nonzero(p_rec)
            max_item = self.n_item - 1
            action = np.min((n_seen, max_item))

        return action
