import numpy as np
from scipy.special import expit


def simulate(
        seed=0,
        sg_u=(0.5, 0.2),
        sg_w=(1.0, 0.4),
        mu=(-4., -2.),
        n_u=20,
        n_w=20,
        n_obs_per_wu=100):

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

    x = np.random.uniform(5, 1000, size=n_obs)
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

    return data, truth
