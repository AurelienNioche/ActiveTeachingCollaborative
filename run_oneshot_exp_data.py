import os
import numpy as np
import pandas as pd

from inference.train import train
from plot.plot_hist_loss import plot_loss
from plot.plot_posterior import plot_posterior

from data_preprocessing.data_preprocessing import get_experimental_data

from inference.flows import NormalizingFlows


BKP_FOLDER = "bkp/exp_data"
FIG_FOLDER = "fig/exp_data"
os.makedirs(BKP_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)


def make_fig(theta_flow, hist_loss):
    print("Making the plots...")
    plot_posterior(
        path=f"{FIG_FOLDER}/posterior.pdf",
        theta_flow=theta_flow)
    plot_loss(
        path=f"{FIG_FOLDER}/loss.pdf",
        hist_loss=hist_loss)


def run_inference(bkp_name="norm_flows",
                  load_bkp=True,
                  n_sample=40,
                  epochs=5000,
                  flow_length=16,
                  optimizer_name="Adam",
                  optimizer_kwargs=None,
                  initial_lr=0.01,
                  seed=123):

    z_bkp_file = f"{BKP_FOLDER}/{bkp_name}_z.p"
    theta_bkp_file = f"{BKP_FOLDER}/{bkp_name}_theta.p"
    hist_bkp_file = f"{BKP_FOLDER}/{bkp_name}_hist_loss.npy"

    if load_bkp:
        try:
            z_flow = NormalizingFlows.load(z_bkp_file)
            theta_flow = NormalizingFlows.load(theta_bkp_file)
            hist_loss = np.load(hist_bkp_file)
            print("Load successfully from backup")
            return z_flow, theta_flow, hist_loss

        except FileNotFoundError:
            print("Didn't find backup. Run the inference process instead...")

    data = get_experimental_data()

    z_flow, theta_flow, hist_loss = train(
        data=data,
        n_sample=n_sample,
        epochs=epochs,
        flow_length=flow_length,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        initial_lr=initial_lr,
        seed=seed)

    z_flow.save(z_bkp_file)
    theta_flow.save(theta_bkp_file)
    np.save(hist_bkp_file, np.asarray(hist_loss))

    return z_flow, theta_flow, hist_loss


def save_population_parameters(theta_flow, batch_size=int(10e5),
                               path="data/param_exp_data.csv"):

    print("Saving population parameters...")

    z0_θ = theta_flow.sample_base_dist(batch_size)
    zk_θ, base_dist_logprob_θ, log_det_θ = theta_flow(z0_θ)

    half_mu1, log_var_u1, log_var_w1 = zk_θ.data[:, :3].T
    half_mu2, log_var_u2, log_var_w2 = zk_θ.data[:, 3:].T

    unconstrained = {
        "mu1": half_mu1.mean().item()*2,
        "sigma_u1": np.exp(0.5 * log_var_u1.mean().item()),
        "sigma_w1": np.exp(0.5 * log_var_w1.mean().item()),
        "mu2": half_mu2.mean().item()*2,
        "sigma_u2": np.exp(0.5 * log_var_u2.mean().item()),
        "sigma_w2": np.exp(0.5 * log_var_w2.mean().item())}

    df_param = pd.DataFrame([unconstrained, ], index=["unconstrained", ])
    df_param.to_csv(path)


def main():

    z_flow, theta_flow, hist_loss = run_inference()
    make_fig(theta_flow, hist_loss)
    save_population_parameters(theta_flow)


if __name__ == "__main__":
    main()
