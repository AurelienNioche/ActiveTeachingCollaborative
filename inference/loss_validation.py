import torch
from torch.nn import BCELoss
import torch.distributions as dist


class LossTeachingValidation(BCELoss):

    def __init__(self):
        super().__init__()

    def __call__(self, z_flow, theta_flow, n_sample,
                 total_n_obs,
                 # total_n_epochs,
                 # epoch,
                 n_u, n_w,
                 u, w, x, r, y):
        # Z: Sample base distribution and apply transformation
        z0_Z = z_flow.sample_base_dist(n_sample)
        zk_Z, ln_q0_Z, sum_ld_Z = z_flow(z0_Z)

        # # θ: Sample base distribution and apply transformation
        # z0_θ = theta_flow.sample_base_dist(n_sample)
        # zk_θ, ln_q0_θ, sum_ld_θ = theta_flow(z0_θ)

        # Get Z-values used for first parameter
        Zu1 = zk_Z[:, :n_u].T
        Zw1 = zk_Z[:, n_u:n_w + n_u].T

        # Get Z-values used for second parameter
        Zu2 = zk_Z[:, n_w + n_u:n_w + n_u * 2].T
        Zw2 = zk_Z[:, n_w + n_u * 2:].T

        # Get Z-values used for second first parameter
        # half_mu1, log_var_u1, log_var_w1 = zk_θ[:, :3].T
        # half_mu2, log_var_u2, log_var_w2 = zk_θ[:, 3:].T

        # Compute Z-values for both parameters
        Z1 = Zu1[u] + Zw1[w]
        Z2 = Zu2[u] + Zw2[w]

        # Go to constrained space
        a = torch.exp(Z1)
        b = torch.sigmoid(Z2)

        # Compute log probability of recall
        log_p = -a * x * (1 - b) ** r

        # Comp. log-likelihood of observations
        ll = dist.Bernoulli(probs=torch.exp(log_p)).log_prob(y).sum(axis=1)

        return super()(torch.exp(ll), y.squeeze())
