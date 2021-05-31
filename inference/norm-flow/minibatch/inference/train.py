import numpy as np
import torch
from torch import optim
from tqdm.autonotebook import tqdm

from . plannar import NormalizingFlow
from . loss import LossTeaching


def train(x, y, r, u, w,
          flow_length=16, epochs=5000, initial_lr=0.01, n_sample=40,
          freq_update_loss=1):

    n_u = len(np.unique(u))
    n_w = len(np.unique(w))

    x = torch.from_numpy(x.reshape(-1, 1))
    y = torch.from_numpy(y.reshape(-1, 1))
    r = torch.from_numpy(r.reshape(-1, 1))

    z_flow = NormalizingFlow(dim=(n_u + n_w) * 2, flow_length=flow_length)
    theta_flow = NormalizingFlow(6, flow_length=flow_length)

    loss_func = LossTeaching()

    optimizer = optim.Adam(
        list(z_flow.parameters()) + list(theta_flow.parameters()),
        lr=initial_lr)

    hist_loss = np.zeros(epochs)

    with tqdm(total=epochs) as pbar:

        for i in range(epochs):

            optimizer.zero_grad()
            loss = loss_func(z_flow=z_flow,
                             theta_flow=theta_flow,
                             n_sample=n_sample,
                             n_u=n_u,
                             n_w=n_w,
                             u=u,
                             w=w,
                             x=x,
                             y=y,
                             r=r)
            loss.backward()
            optimizer.step()

            hist_loss[i] = loss.item()

            if i % freq_update_loss == 0 and i > 0:
                pbar.set_postfix(
                    {'loss': np.mean(hist_loss[i - freq_update_loss:i])})
            pbar.update()

    return z_flow, theta_flow, hist_loss


