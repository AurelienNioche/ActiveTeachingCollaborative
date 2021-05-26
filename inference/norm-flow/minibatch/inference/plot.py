import os

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_loss(hist_loss, zoom_start=1000, name=""):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    ax = axes[0]
    ax.plot(hist_loss)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    ax = axes[1]
    ax.plot(np.arange(len(hist_loss))[-zoom_start:], hist_loss[-zoom_start:])
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    fig.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}_loss.pdf"))


def plot_posterior(theta_flow, batch_size=1000, truth=None,
                   name=""):

    z0_θ = theta_flow.sample_prior(batch_size)
    zk_θ, prior_logprob_θ, log_det_θ = theta_flow(z0_θ)

    mu1, log_var_u1, log_var_w1 = zk_θ.data[:, :3].T
    mu2, log_var_u2, log_var_w2 = zk_θ.data[:, 3:].T

    fig, axes = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)

    alpha = 0.1

    ax = axes[0, 0]
    ax.set_title(r"$\alpha$ - $U$")
    ax.scatter(mu1, log_var_u1,
               color='C0', alpha=alpha, label=r"$\theta \sim Q(\theta)$")
    ax.set_xlabel("$\mu /2$")
    ax.set_ylabel("$log(\sigma^2)$")

    if truth is not None:
        ax.scatter(truth['mu'][0] / 2, np.log(truth['sg_u'][0] ** 2),
                   edgecolors='C3', alpha=1,
                   label='theoretical', facecolors='none', s=40)
        ax.scatter(truth['mu_smp'][0] / 2, np.log(truth['sg_u_smp'][0] ** 2),
                   facecolors='none', edgecolors='C3', alpha=1, s=80,
                   linewidths=3, label="data")
    ax.legend()

    ax = axes[1, 0]
    ax.set_title(r"$\alpha$ - $W$")
    ax.scatter(mu1, log_var_w1,
               color='C0', alpha=alpha, label=r"$\theta \sim Q(\theta)$")
    ax.set_xlabel("$\mu /2$")
    ax.set_ylabel("$log(\sigma^2)$")
    if truth is not None:
        ax.scatter(truth['mu'][0] / 2, np.log(truth['sg_w'][0] ** 2),
                   edgecolors='C3',
                   label='theoretical', facecolors='none', s=40)
        ax.scatter(truth['mu_smp'][0] / 2, np.log(truth['sg_w_smp'][0] ** 2),
                   facecolors='none', edgecolors='C3', alpha=1, s=80,
                   linewidths=3, label="data")
    ax.legend()

    ax = axes[0, 1]
    ax.set_title(r"$\beta$ - $U$")
    ax.scatter(mu2, log_var_u2,
               color='C1', alpha=alpha, label=r"$\theta \sim Q(\theta)$")

    ax.set_xlabel("$\mu /2$")
    ax.set_ylabel("$log(\sigma^2)$")
    if truth is not None:
        ax.scatter(truth['mu'][1] / 2, np.log(truth['sg_u'][1] ** 2),
                   edgecolors='C3',
                   label='theoretical', facecolors='none', s=40)
        ax.scatter(truth['mu_smp'][1] / 2, np.log(truth['sg_u_smp'][1] ** 2),
                   facecolors='none', edgecolors='C3', s=80,
                   linewidths=3, label="data")
    ax.legend()

    ax = axes[1, 1]
    ax.set_title(r"$\beta$ - $W$")
    ax.scatter(mu2, log_var_w2,
               color='C1', alpha=alpha, label=r"$\theta \sim Q(\theta)$")
    ax.set_xlabel("$\mu /2$")
    ax.set_ylabel("$log(\sigma^2)$")
    if truth is not None:
        ax.scatter(truth['mu'][1] / 2, np.log(truth['sg_w'][1] ** 2),
                   edgecolors='C3', alpha=1,
                   label='theoretical', facecolors='none', s=40)
        ax.scatter(truth['mu_smp'][1] / 2, np.log(truth['sg_w_smp'][1] ** 2),
                   facecolors='none', edgecolors='C3', alpha=1, s=80,
                   linewidths=3, label="data")
    ax.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}_posterior.pdf"))
