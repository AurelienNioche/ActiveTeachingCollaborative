import os
import numpy as np
import torch

from inference.train import train
from inference.flows import NormalizingFlows
from simulate.simulate import simulate
from plot.plot_hist_loss import plot_loss
from plot.plot_posterior import plot_posterior

BKP_FOLDER = 'bkp/artificial'
FIG_FOLDER = 'fig/artificial'
os.makedirs(BKP_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)


SEED_DATA_GENERATION = 0


def make_fig(theta_flow, hist_loss, truth):
    print("Making the plots...")
    plot_posterior(
        path=f"{FIG_FOLDER}/posterior.pdf",
        theta_flow=theta_flow,
        truth=truth)
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
    truth_bkp_file = f"{BKP_FOLDER}/{bkp_name}_truth.p"

    if load_bkp:
        try:
            z_flow = NormalizingFlows.load(z_bkp_file)
            theta_flow = NormalizingFlows.load(theta_bkp_file)
            hist_loss = np.load(hist_bkp_file)
            truth = torch.load(truth_bkp_file)
            print("Load successfully from backup")
            return z_flow, theta_flow, hist_loss, truth

        except FileNotFoundError:
            print("Didn't find backup. Run the inference process instead...")

    data, truth = simulate(use_torch=True, seed=SEED_DATA_GENERATION)

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
    np.save(file=hist_bkp_file, arr=np.asarray(hist_loss))
    torch.save(obj=truth, f=truth_bkp_file)

    return z_flow, theta_flow, hist_loss, truth


def main():

    z_flow, theta_flow, hist_loss, truth = run_inference(load_bkp=False)
    make_fig(theta_flow=theta_flow, hist_loss=hist_loss, truth=truth)


if __name__ == "__main__":
    main()
