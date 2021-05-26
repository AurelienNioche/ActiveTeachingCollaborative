import torch
import torch.distributions as dist
from torch import nn


class LossTeaching(nn.Module):

    """
    Loss function for the following model:

    \begin{align}
    Z_u^\rho &\sim \mathcal{N}(0, \sigma_u^\rho)\\
    Z_w^\rho &\sim \mathcal{N}(0, \sigma_w^\rho) \\
    Z_{u, w}^\rho &= \mu^\rho + Z_u^\rho + Z_w^\rho \\
    \end{align}
    where $Z_u^{\rho}$ is a random variable whose distribution is specific to user $u$ and parameter $\rho$, and $\rho \in {\alpha, \beta}$.

    The probability of recall for user $u$ and item/word $w$ at time $t$ is defined as:
    \begin{align}
    p(\omega = 1 \mid t, u, w) &= e^{-Z_{u, w}^\alpha (1-Z_{u, w}^\beta)^n \delta_{u, w}^t}  \\
    \end{align}
    where $\delta_{u, w}^t$ is the time elapsed since the last presentation for user $u$, item $w$ at time $t$.
    """

    def __init__(self, x, y, r, u, w, n_u, n_w):
        super().__init__()
        self.x = torch.from_numpy(x.reshape(-1, 1))
        self.y = torch.from_numpy(y.reshape(-1, 1))
        self.r = torch.from_numpy(r.reshape(-1, 1))
        self.u = u
        self.w = w
        self.n_u = n_u
        self.n_w = n_w

    def forward(self, z_flow, theta_flow, batch_size):
        z0_Z = z_flow.sample_prior(batch_size)
        zk_Z, prior_logprob_Z, log_det_Z = z_flow(z0_Z)

        z0_θ = theta_flow.sample_prior(batch_size)
        zk_θ, prior_logprob_θ, log_det_θ = theta_flow(z0_θ)

        ln_q0_Z = prior_logprob_Z.sum()
        ln_q0_θ = prior_logprob_θ.sum()

        sum_ld_Z = log_det_Z.sum()
        sum_ld_θ = log_det_θ.sum()

        Zu1 = zk_Z[:, :self.n_u].T
        Zw1 = zk_Z[:, self.n_u:self.n_w + self.n_u].T

        Zu2 = zk_Z[:, self.n_w + self.n_u:self.n_w + self.n_u * 2].T
        Zw2 = zk_Z[:, self.n_w + self.n_u * 2:].T

        mu1, log_var_u1, log_var_w1 = zk_θ[:, :3].T
        mu2, log_var_u2, log_var_w2 = zk_θ[:, 3:].T

        Z1 = Zu1[self.u] + Zw1[self.w]
        Z2 = Zu2[self.u] + Zw2[self.w]

        a = torch.exp(Z1)
        b = torch.sigmoid(Z2)

        log_p = -a * self.x * (1 - b) ** self.r

        ll = dist.Bernoulli(probs=torch.exp(log_p)).log_prob(self.y).sum()

        ll_Zu1 = dist.Normal(mu1, torch.exp(0.5 * log_var_u1)).log_prob(
            Zu1).sum()
        ll_Zw1 = dist.Normal(mu1, torch.exp(0.5 * log_var_w1)).log_prob(
            Zw1).sum()

        ll_Zu2 = dist.Normal(mu2, torch.exp(0.5 * log_var_u2)).log_prob(
            Zu2).sum()
        ll_Zw2 = dist.Normal(mu2, torch.exp(0.5 * log_var_w2)).log_prob(
            Zw2).sum()

        to_min = (ln_q0_Z + ln_q0_θ - sum_ld_Z - sum_ld_θ
                  - ll - ll_Zu1 - ll_Zu2 - ll_Zw1 - ll_Zw2) / batch_size
        return to_min