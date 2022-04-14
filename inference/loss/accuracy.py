import numpy as np
import torch
from torch import distributions as dist


class AccuracyTeaching:
    @staticmethod
    def __call__(z_flow, theta_flow, n_sample,
                 total_n_obs,
                 # total_n_epochs,
                 # epoch,
                 n_u, n_w,
                 u, w, x, r, y,
                 **kwargs):

        # Z: Sample base distribution and apply transformation
        z0_Z = z_flow.sample_base_dist(n_sample)
        zk_Z, ln_q0_Z, sum_ld_Z = z_flow(z0_Z)

        # Get Z-values used for first parameter
        Zu1 = zk_Z[:, :n_u].T
        Zw1 = zk_Z[:, n_u:n_w + n_u].T

        # Get Z-values used for second first parameter
        Zu2 = zk_Z[:, n_w + n_u:n_w + n_u * 2].T
        Zw2 = zk_Z[:, n_w + n_u * 2:].T

        # Compute Z-values for both parameters
        Z1 = Zu1[u] + Zw1[w]
        Z2 = Zu2[u] + Zw2[w]

        # Go to constrained space
        param1 = torch.exp(Z1)
        param2 = torch.sigmoid(Z2)

        # Compute log probability of recall
        log_p = -param1 * x * (1 - param2) ** r

        p = torch.exp(log_p)
        y_pred = (p > 0.5).float()
        accuracy = (y == y_pred).float().sum() / y.size(0)

        return accuracy