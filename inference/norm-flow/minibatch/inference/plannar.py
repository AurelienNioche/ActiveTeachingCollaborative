import os

import torch
import torch.distributions as dist
from torch import nn
import torch.nn.functional as F

from . utils import safe_log


class NormalizingFlow(nn.Module):
    """
    Adapted from https://github.com/ex4sperans/variational-inference-with-normalizing-flows

    [Rezende and Mohamed, 2015]

    """

    BKP_DIR = "bkp"

    def __init__(self, dim, flow_length):
        super().__init__()

        self.transforms = nn.Sequential(*(
            PlanarFlow(dim) for _ in range(flow_length)
        ))

        self.log_jacobians = nn.Sequential(*(
            PlanarFlowLogDetJacobian(t) for t in self.transforms
        ))

        self.dim = dim
        self.flow_length = flow_length

        self.mu = nn.Parameter(torch.zeros(dim).uniform_(-0.01, 0.01))
        self.log_var = nn.Parameter(torch.zeros(dim).uniform_(-0.01, 0.01))

    def sample_base_dist(self, batch_size):
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn((batch_size, self.dim))
        return self.mu + eps * std

    def log_prob_base_dist(self, x):
        std = torch.exp(0.5 * self.log_var)
        return dist.Normal(self.mu, std).log_prob(x).sum(axis=-1)

    def forward(self, x):
        batch_size = x.shape[0]

        log_prob_base_dist = self.log_prob_base_dist(x)

        log_jacobians = torch.zeros((self.flow_length, batch_size, 1))

        for i in range(self.flow_length):
            log_jacobians[i] = self.log_jacobians[i](x)
            x = self.transforms[i](x)

        return x, log_prob_base_dist, log_jacobians

    def save(self, name):
        path = os.path.join(self.BKP_DIR, name)
        os.makedirs(self.BKP_DIR, exist_ok=True)
        torch.save(self.state_dict(),
                   path+"_state_dict.p")
        torch.save(dict(dim=self.dim, flow_length=self.flow_length),
                   path+"_attr.p")

    @classmethod
    def load(cls, name):
        path = os.path.join(cls.BKP_DIR, name)
        model = cls(**torch.load(path+"_attr.p"))
        model.load_state_dict(torch.load(path+"_state_dict.p"))
        return model


class PlanarFlow(nn.Module):
    """
    Planar flow.

        z = f(x) = x + u h(wᵀx + b)
    """

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        activation = F.linear(x, self.weight, self.bias)
        return x + self.scale * self.tanh(activation)


class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, x):
        activation = F.linear(x, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return safe_log(det_grad.abs())