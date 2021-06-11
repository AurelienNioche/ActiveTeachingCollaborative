import os

import torch
import torch.distributions as dist
from torch import nn
import torch.nn.functional as F


def safe_log(z):
    return torch.log(z + 1e-7)


class NormalizingFlow(nn.Module):
    """
    Adapted from https://github.com/ex4sperans/variational-inference-with-normalizing-flows

    [Rezende and Mohamed, 2015]

    """

    BKP_DIR = "bkp"

    def __init__(self, dim, flow_length, flow_model=None):
        super().__init__()
        if flow_model is None:
            flow_model = PlanarFlow

        self.transforms = nn.Sequential(*(
            flow_model(dim) for _ in range(flow_length)
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

        log_prob_base_dist = self.log_prob_base_dist(x)

        log_det = torch.zeros(x.shape[0])

        for i in range(self.flow_length):
            x, ld = self.transforms[i](x)
            log_det += ld

        return x, log_prob_base_dist, log_det

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

        self.weight = nn.Parameter(torch.empty(1, dim))
        self.bias = nn.Parameter(torch.empty(1))
        self.scale = nn.Parameter(torch.empty(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.uniform_(self.weight, -0.01, 0.01)
        nn.init.uniform_(self.scale, -0.01, 0.01)
        nn.init.uniform_(self.bias, -0.01, 0.01)

    def forward(self, x):
        activation = F.linear(x, self.weight, self.bias)
        out = self.tanh(activation)

        f = x + self.scale * out

        psi = (1 - out ** 2) * self.weight
        det_grad = 1 + psi@self.scale.T
        ld = safe_log(det_grad.squeeze(-1).abs())

        return f, ld


class RadialFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.z0 = nn.Parameter(torch.empty(dim))
        self.log_alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))

        self.dim = dim

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.uniform_(self.z0, -0.01, 0.01)
        nn.init.uniform_(self.log_alpha, -0.01, 0.01)
        nn.init.uniform_(self.beta, -0.01, 0.01)

    def forward(self, x):

        z_sub = x - self.z0
        alpha = torch.exp(self.log_alpha)
        r = torch.norm(z_sub)
        h = 1 / (alpha + r)
        f = x + self.beta * h * z_sub
        log_det = (self.dim - 1) * safe_log(1 + self.beta * h) + \
            safe_log(1 + self.beta * h + self.beta - self.beta * r / (alpha + r) ** 2)

        return f, log_det
