import os
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from . flows import NormalizingFlows
from . loss.bce import BCETeaching
from . loss.elbo import ELBOTeaching
from . loss.accuracy import AccuracyTeaching


def train(
        dataset,
        bkp_folder,
        bkp_name,
        load_if_exists=False,
        batch_size=None,
        training_split=1.0,
        flow_length=16,
        epochs=5000,
        optimizer_name="Adam",
        optimizer_kwargs=None,
        initial_lr=0.01,
        constant_lr=False,
        scheduler_name=None,
        n_sample=40,
        seed=123,
        truth=None,  # just for putting all backup at same place
        ):

    z_bkp_file = f"{bkp_folder}/{bkp_name}_z.p"
    theta_bkp_file = f"{bkp_folder}/{bkp_name}_theta.p"
    hist_loss_bkp_file = f"{bkp_folder}/{bkp_name}_hist_loss.npy"
    hist_val_bkp_file = f"{bkp_folder}/{bkp_name}_hist_val.npy"
    hist_train_bkp_file = f"{bkp_folder}/{bkp_name}_hist_train.npy"
    truth_bkp_file = f"{bkp_folder}/{bkp_name}_truth.p"

    if load_if_exists:
        try:
            z_flow = NormalizingFlows.load(z_bkp_file)
            theta_flow = NormalizingFlows.load(theta_bkp_file)
            hist_loss = np.load(hist_loss_bkp_file)
            hist_val = torch.load(hist_val_bkp_file)
            hist_train = torch.load(hist_train_bkp_file)
            if truth is not None:
                truth = torch.load(truth_bkp_file)
            print("Load successfully from backup")
            return z_flow, theta_flow, hist_loss, hist_val, hist_train

        except FileNotFoundError:
            print("Didn't find backup. Run the inference process instead...")

    np.random.seed(seed)
    torch.manual_seed(seed)

    n = len(dataset)

    if batch_size is None:
        batch_size = n

    if training_split < 1.0:
        n_training = int(training_split * n)
        n_validation = n - n_training

        train_set, val_set = random_split(
            dataset,
            [n_training, n_validation])

        print("N training", n_training)
        print("N validation", n_validation)

        training_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        validation_data = DataLoader(val_set, batch_size=n_validation, shuffle=True)
    else:
        training_data = DataLoader(dataset, batch_size=n, shuffle=False)
        n_training = n
        n_validation = 0
        validation_data = ()

    n_u, n_w = dataset.n_u, dataset.n_w

    z_flow = NormalizingFlows(dim=(n_u + n_w) * 2, flow_length=flow_length)
    theta_flow = NormalizingFlows(6, flow_length=flow_length)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = getattr(optim, optimizer_name)(
            list(z_flow.parameters()) + list(theta_flow.parameters()),
            lr=initial_lr, **optimizer_kwargs)

    if scheduler_name is not None:
        scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer)
        assert constant_lr is False
    else:
        scheduler = None

    hist_loss = []

    loss_func = ELBOTeaching()

    metrics = {
        "bce": BCETeaching(),
        "elbo": ELBOTeaching(),
        "accuracy": AccuracyTeaching()
    }

    hist_val = {k: [] for k in metrics}
    hist_train = {k: [] for k in metrics}

    with tqdm(total=epochs) as pbar:

        for i in range(epochs):

            theta_flow.train()
            z_flow.train()

            for d in training_data:

                optimizer.zero_grad()

                loss = loss_func(z_flow=z_flow,
                                 theta_flow=theta_flow,
                                 n_sample=n_sample,
                                 n_u=n_u,
                                 n_w=n_w,
                                 total_n_obs=n_training,
                                 **d)

                loss.backward()
                optimizer.step()

                hist_loss.append(loss.item())

                if i > 0:
                    pbar.set_postfix({'loss': hist_loss[-1]})

                if constant_lr:
                    for g in optimizer.param_groups:
                        g['lr'] = initial_lr

                if scheduler is not None:
                    scheduler.step()

            pbar.update()

            z_flow.eval()
            theta_flow.eval()

            with torch.no_grad():
                val = {k: [] for k in metrics}
                for d in validation_data:
                    for k, func in metrics.items():
                        val[k].append(func(
                            z_flow=z_flow,
                            n_sample=n_sample,
                            n_u=n_u,
                            n_w=n_w,
                            **d).item())

                val = {k: [] for k in metrics}
                for d in validation_data:
                    for k, func in metrics.items():
                        val[k].append(func(
                            z_flow=z_flow,
                            theta_flow=theta_flow,
                            n_sample=n_sample,
                            n_u=n_u,
                            n_w=n_w,
                            total_n_obs=n_validation,
                            **d).item())

                for k in metrics:
                    if len(val[k]):
                        hist_val[k].append(np.mean(val[k]))

                val = {k: [] for k in metrics}
                for d in training_data:
                    for k, func in metrics.items():
                        val[k].append(func(
                            z_flow=z_flow,
                            theta_flow=theta_flow,
                            n_sample=n_sample,
                            n_u=n_u,
                            n_w=n_w,
                            total_n_obs=n_training,
                            **d).item())

                for k in metrics:
                    hist_train[k].append(np.mean(val[k]))

    os.makedirs(bkp_folder, exist_ok=True)

    z_flow.save(z_bkp_file)
    theta_flow.save(theta_bkp_file)
    np.save(file=hist_loss_bkp_file, arr=np.asarray(hist_loss))
    torch.save(f=hist_train_bkp_file, obj=hist_train)
    torch.save(f=hist_val_bkp_file, obj=hist_val)

    if truth is not None:
        torch.save(obj=truth, f=truth_bkp_file)

    return z_flow, theta_flow, hist_loss, hist_val, hist_train


