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


def train_minibatch_schedule(
        data,
        n_epochs=5000,
        initial_lr=0.01,
        flow_length=16,
        n_sample=40,
        batch_size=None,
        optimizer_name="Adam",
        optimizer_kwargs=None,
        use_scheduler=False,
        scheduler_name=None,
        constant_lr=False,
        seed=123,
        online_plot=False,  # Plot online losses
        online_plot_freq_update=None,  # only matters if online_plot is true
        online_plot_max_n=None,  # only matters if online_plot is true
        clip_gradients=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_u = len(np.unique(data['u']))
    n_w = len(np.unique(data['w']))
    n_obs = len(data['u'])

    if batch_size is None:
        batch_size = n_obs

    z_flow = NormalizingFlows(dim=(n_u + n_w) * 2, flow_length=flow_length)
    theta_flow = NormalizingFlows(6, flow_length=flow_length)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer = getattr(torch.optim, optimizer_name)(
        list(z_flow.parameters()) + list(theta_flow.parameters()),
        lr=initial_lr, **optimizer_kwargs)

    # if scheduler_name is not None:
    #    scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer)
    #    assert constant_lr is False
    # else:
    #    scheduler = None
    if use_scheduler is True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    else:
        scheduler = None

    loss_func = LossMinibatch()

    indices = np.arange(n_obs)

    with LossTracker(
            total=n_epochs,
            online_plot=online_plot,
            online_plot_freq_update=online_plot_freq_update,
            online_plot_max_n=online_plot_max_n) as loss_tracker:

        for epoch in range(n_epochs):

            np.random.shuffle(indices)

            for i in range(0, n_obs, batch_size):

                idx_minibatch = indices[i:i + batch_size]

                d = {k: v[idx_minibatch] for k, v in data.items()}

                optimizer.zero_grad()
                loss = loss_func(z_flow=z_flow,
                                 theta_flow=theta_flow,
                                 n_sample=n_sample,
                                 n_u=n_u, n_w=n_w,
                                 total_n_obs=n_obs,
                                 **d)
                loss.backward()

                if clip_gradients:
                    torch.nn.utils.clip_grad_norm(list(z_flow.parameters()) + list(theta_flow.parameters()),
                                                  max_norm=1)

                optimizer.step()

                loss_tracker.append(loss.item())

                if constant_lr:
                    for g in optimizer.param_groups:
                        g['lr'] = initial_lr

            if scheduler is not None:
                scheduler.step(loss)

            loss_tracker.update()

    return z_flow, theta_flow, loss_tracker.hist_loss



def objective(trial, data):
    try:
        flow_length = trial.suggest_int("flow_length", 1, 50)
        n_sample = trial.suggest_int("n_sample", 1, 100)

        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        if optimizer_name == "Adam":
            optimizer_kwargs = dict(
                betas=(trial.suggest_float("Adam_beta1", 0.8, 1.0), trial.suggest_float("Adam_beta2", 0.8, 1.0)),
                weight_decay=trial.suggest_float("Adam_weight_decay", 0.0, 0.1),
                amsgrad=trial.suggest_categorical("Adam_amsgrad", [False, True]),
                eps=1e-08)

        elif optimizer_name == "RMSprop":
            optimizer_kwargs = dict(
                alpha=trial.suggest_float("alpha", 0, 1),
                centered=trial.suggest_categorical("RMSprop_centered", [False, True]),
                weight_decay=trial.suggest_float("RMSprop_weight_decay", 0.0, 0.1),
                momentum=trial.suggest_float("RMSprop_momentum", 0.0, 0.1),
                eps=1e-08)

        use_scheduler = False  # trial.suggest_categorical("use_schedule", [True, False])
        # ReduceLROnPlateau
        # if use_schedule:

        kwargs = dict(
            n_epochs=200,
            initial_lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            use_scheduler=use_scheduler,
            flow_length=flow_length,
            n_sample=n_sample,
            batch_size=1000
        )

        z_flow, theta_flow, hist_loss = train_minibatch_schedule(data, **kwargs)

        loss = LossTeaching()(
            z_flow=z_flow,
            theta_flow=theta_flow,
            n_sample=100,
            n_u=len(np.unique(data['u'])),
            n_w=len(np.unique(data['w'])),
            **data).item()
    except Exception as e:
        raise (e)
        loss = np.nan
    return loss



import logging
import sys


def main():

    dataset, truth = simulate(use_torch=True, seed=123, use_torch_dataset=True)

    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "optim_hyperparam_2"  # Unique identifier of the study.
    storage_name = f"sqlite:///bkp/{study_name}.db"

    study = optuna.create_study(direction="minimize", study_name=study_name,
                                storage=storage_name, load_if_exists=False)
    study.optimize(lambda trial: objective(trial, data), n_trials=300)

    study.best_params

    fig = optuna.visualization.plot_optimization_history(study)
    fig

    fig.update_yaxes(range=[8000, 10000])

    optuna.visualization.plot_intermediate_values(study)

    optuna.visualization.plot_parallel_coordinate(study)

    optuna.visualization.plot_contour(study)

    fig = optuna.visualization.plot_slice(study)
    fig

    fig.update_yaxes(range=[8000, 10000])
    fig

    optuna.visualization.plot_param_importances(study)

    optuna.visualization.plot_edf(study)

