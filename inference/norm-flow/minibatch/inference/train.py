import numpy as np
from torch import optim
from tqdm.autonotebook import tqdm

from . plannar import NormalizingFlow
from . loss import LossTeaching


def train(x, y, r, u, w, n_u, n_w,
          flow_length=16, epochs=10000, initial_lr=0.01, batch_size=40,
          freq_update_loss=50):

    z_flow = NormalizingFlow(dim=(n_u + n_w) * 2, flow_length=flow_length)
    theta_flow = NormalizingFlow(6, flow_length=flow_length)

    loss_func = LossTeaching(x=x, y=y, r=r, u=u, w=w, n_u=n_u, n_w=n_w)

    optimizer = optim.Adam(
        list(z_flow.parameters()) + list(theta_flow.parameters()),
        lr=initial_lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

    hist_loss = np.zeros(epochs)

    with tqdm(total=epochs) as pbar:

        for i in range(epochs):

            optimizer.zero_grad()
            loss = loss_func(z_flow=z_flow,
                             theta_flow=theta_flow,
                             batch_size=batch_size)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            hist_loss[i] = loss.item()

            if i % freq_update_loss == 0 and i > 0:
                pbar.set_postfix(
                    {'loss': np.mean(hist_loss[i - freq_update_loss:i])})
            pbar.update()

    return z_flow, theta_flow, hist_loss


