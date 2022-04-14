import numpy as np
from scipy.special import expit
import torch

from torch.utils.data import Dataset


def simulate(
        seed,
        sg_u=(0.5, 0.2),
        sg_w=(1.0, 0.4),
        mu=(-4., -2.),
        min_x=5,
        max_x=1000,
        n_u=20,
        n_w=20,
        n_obs_per_wu=100,
        use_torch=False,
        use_torch_dataset=False):

    np.random.seed(seed)
    n_obs = n_obs_per_wu * n_u * n_w

    u = np.concatenate(
        [np.zeros(n_obs_per_wu * n_w, dtype=int) + i for i in range(n_u)])
    w = np.tile(np.concatenate(
        [np.zeros(n_obs_per_wu, dtype=int) + i for i in range(n_w)]), n_u)

    r = np.zeros(n_obs)
    r[:] = np.tile(np.concatenate(
        [np.arange(1, n_obs_per_wu + 1, dtype=int) for _ in range(n_w)]), n_u)

    Zu = np.random.normal(np.zeros(2), sg_u, size=(n_u, 2))
    Zw = np.random.normal(np.zeros(2), sg_w, size=(n_w, 2))

    x = np.random.uniform(min_x, max_x, size=n_obs)
    rd = np.random.random(size=n_obs)

    y = np.zeros(shape=n_obs)

    Z = mu + Zu[u] + Zw[w]

    a = np.exp(Z[:, 0])
    b = expit(Z[:, 1])
    neg_rate = - a * x * (1 - b) ** r
    p = np.exp(neg_rate)
    y[:] = p > rd

    data = {'u': u, 'w': w,
            'x': x, 'r': r,
            'y': y}

    sg_u_smp = np.std(Zu, axis=0)
    sg_w_smp = np.std(Zw, axis=0)
    mu_smp = np.mean(Z, axis=0)
    truth = {'mu': mu, 'sg_u': sg_u, 'sg_w': sg_w,
             'mu_smp': mu_smp, 'sg_u_smp': sg_u_smp, 'sg_w_smp': sg_w_smp}

    if use_torch or use_torch_dataset:
        data = {
            'x': torch.from_numpy(data['x'].reshape(-1, 1)),
            'y': torch.from_numpy(data['y'].reshape(-1, 1)),
            'r': torch.from_numpy(data['r'].reshape(-1, 1)),
            'u': data['u'],
            'w': data['w']}

    # extra step to have a dataset object
    if use_torch_dataset:
        data = TeachingDataset(**data)
        # dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    return data, truth


class TeachingDataset(Dataset):
    def __init__(self, u, w, x, r, y):
        super().__init__()
        self.x = x  # torch.from_numpy(x.reshape(-1, 1))
        self.y = y  # torch.from_numpy(y.reshape(-1, 1))
        self.r = r  # torch.from_numpy(r.reshape(-1, 1))
        self.u = u
        self.w = w

        self.n_u = len(np.unique(u))
        self.n_w = len(np.unique(w))
        self.n_obs = len(self.u)

    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx):
        return {'x': self.x[idx],
                'y': self.y[idx],
                'r': self.r[idx],
                'u': self.u[idx],
                'w': self.w[idx]}
