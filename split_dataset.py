import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import optuna
import pickle
import torch

from simulate.simulate import simulate

from inference.loss import LossTeaching
from inference.train_minibatch import train_minibatch
from inference.flows import NormalizingFlows
from inference.loss_minibatch import LossMinibatch

from plot.plot_hist_loss import plot_loss
from plot.plot_posterior import plot_posterior
from plot.loss_tracker import LossTracker

from run_oneshot_artificial import run_inference, SEED_DATA_GENERATION


def main():

    dataset, truth = simulate(use_torch=True, seed=SEED_DATA_GENERATION,
                              use_torch_dataset=True)
    n = len(dataset)

    prop_training = 0.8
    n_training = int(prop_training*n)

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_training, n - n_training])

    z_flow, theta_flow, hist_loss = run_inference(
        data=train_set,
        truth=truth,
        load_bkp=False,
        bkp_name="norm_flow_split")




if __name__ == "__main__":
    main()
