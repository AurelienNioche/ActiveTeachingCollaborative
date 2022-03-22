from abc import ABC

import numpy as np
import torch
from torch import optim
from torch import distributions as dist

import matplotlib.pyplot as plt

import seaborn as sns
import pytorch_lightning as pl   # Equivalent of Keras but for Pytorch

from inference.flows.flows import NormalizingFlows

sns.set()


def plot_results(mu_samples, log_var_samples, 
                 obs_mu, obs_sigma, 
                 true_mu, true_sigma,
                 losses,
                 alpha_scatter=0.3, 
                 alpha_pdf=0.1):
    
    true_log_var = np.log(true_sigma**2)
    obs_log_var = np.log(obs_sigma**2)
    
    sigma_samples = torch.exp(0.5*log_var_samples)
    
    fig, axes = plt.subplots(ncols=3, figsize=(20, 8))

    ax = axes[0]
    ax.set_title("Unconstrained space")
    ax.scatter(mu_samples, log_var_samples, alpha=alpha_scatter, color="C1")
    ax.scatter(true_mu, true_log_var, label="theoretic", facecolors='none', edgecolors='C0', s=80,
               linewidths=1,)
    ax.scatter(obs_mu, obs_log_var, label="observed", facecolors='none', edgecolors='C0', s=80,
               linewidths=3,)
    ax.set_xlabel("$\mu$")
    ax.set_ylabel("$\ln \sigma^2$")
    ax.legend()

    ax = axes[1]
    x = torch.linspace(-2, 10, 100).unsqueeze(0)
    
    pdf_true = dist.Normal(true_mu, true_sigma).log_prob(x).exp()
    pdf_sample = dist.Normal(mu_samples, sigma_samples).log_prob(x.unsqueeze(-1)).exp()

    ax.plot(x.squeeze(), pdf_sample.squeeze(), color="C1", alpha=alpha_pdf, label='inferred')
    ax.plot(x.squeeze(), pdf_true.squeeze(), color='C0', label="theoretic")
    # ax.hist(observations.data.numpy(), bins='auto', alpha=0.3, density=True, 
    #         color='C0', label="observed", zorder=1)

    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels,)
    ax.set_xlim(torch.min(x), torch.max(x))

    ax = axes[2]
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    plt.tight_layout()
    plt.show()


class Model(pl.LightningModule, ABC):
    
    def __init__(self, n_dim, flow_length,
                 max_epochs,
                 total_n_obs,
                 n_sample,
                 init_lr):
        
        super().__init__()
        self.flow = NormalizingFlows(dim=n_dim, flow_length=flow_length)
        self.init_lr = init_lr
        self.n_sample = n_sample
        self.total_n_obs = total_n_obs
        self.max_epochs = max_epochs

        self.n_round = 0
        self.losses = []
        
    def configure_optimizers(self):
        
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.init_lr)
        # optimizer = optim.RMSprop(self.parameters(), lr=self.init_lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        
        x, _ = batch
        
        z0 = self.flow.sample_base_dist(self.n_sample)
        zk, log_q0, log_sum_det = self.flow(z0)  # shape: (n_sample, n_dim), (n_sample, ) (n_sample, )
    
        mu, log_var = zk.T
        sigma = torch.exp(0.5 * log_var)
        lls = dist.Normal(mu, sigma).log_prob(x).sum(axis=0)  # shape: (n_obs, n_sample) => (n_sample, )

        if batch_idx == 0:
            self.n_round += 1

        batch_ratio = self.total_n_obs / x.size(0)
        beta = min(batch_ratio, 0.01 + batch_ratio*self.n_round/self.max_epochs)  # 0.01  # self.total_n_obs / x.size(0)
        
        loss = (log_q0 - log_sum_det - beta*lls).sum() \
               / (self.total_n_obs * self.n_sample)
        
        self.log('loss_epoch', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.losses.append(loss.item())
        return loss
    
    def generate_samples(self, n_sample):
        
        with torch.no_grad():
            z0 = self.flow.sample_base_dist(n_sample)  
            zk, prior_logprob, log_det = self.flow(z0)
        mu_samples, log_var_samples = zk.T
        
        return mu_samples, log_var_samples


def generate_dataset(mu, sigma, n_obs, seed):
    torch.manual_seed(seed)
    obs = dist.Normal(loc=mu, scale=sigma).sample((n_obs, ))

    obs_mu = torch.mean(obs).item()
    obs_sigma = torch.std(obs).item()
    print(f"obs mu = {obs_mu:.2f}")
    print(f"obs sigma = {obs_sigma:.2f}")

    fig, ax = plt.subplots()
    ax.hist(obs.data.numpy(), bins='auto')
    plt.show()

    # # Model
    X_train = obs.unsqueeze(-1)
    y_train = torch.zeros(obs.size(0))  # This will be trashed

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    return dataset, obs_mu, obs_sigma


def run_minibatch(dataset, max_epochs, batch_size):

    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=0)

    model = Model(n_dim=2, flow_length=16,
                  max_epochs=max_epochs,
                  n_sample=40,
                  init_lr=0.01,
                  total_n_obs=len(dataset))
    trainer = pl.Trainer(max_epochs=model.max_epochs)
    trainer.fit(model, trainloader)

    return model


def run_oneshot(dataset, max_epochs):

    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=len(dataset),
                                              num_workers=0)

    model = Model(n_dim=2, flow_length=16,
                  n_sample=40,
                  max_epochs=max_epochs,
                  init_lr=0.01,
                  total_n_obs=len(dataset))
    trainer = pl.Trainer(max_epochs=model.max_epochs)
    trainer.fit(model, trainloader)

    return model


def main():

    n_obs = 200
    true_mu = 4
    true_sigma = 2
    seed = 1234

    dataset, obs_mu, obs_sigma \
        = generate_dataset(mu=true_mu, sigma=true_sigma, n_obs=n_obs, seed=seed)

    # Oneshot for comparison
    model = run_oneshot(dataset, max_epochs=500)

    mu_samples, log_var_samples = model.generate_samples(n_sample=500)
    plot_results(
        true_mu=true_mu, true_sigma=true_sigma,
        obs_mu=obs_mu, obs_sigma=obs_sigma,
        mu_samples=mu_samples, log_var_samples=log_var_samples,
        losses=model.losses)

    # Minibatch
    model = run_minibatch(dataset, max_epochs=500, batch_size=32)

    mu_samples, log_var_samples = model.generate_samples(n_sample=500)
    plot_results(
        true_mu=true_mu, true_sigma=true_sigma,
        obs_mu=obs_mu, obs_sigma=obs_sigma,
        mu_samples=mu_samples, log_var_samples=log_var_samples,
        losses=model.losses)


if __name__ == "__main__":
    main()





