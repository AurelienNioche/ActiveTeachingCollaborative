import os
import matplotlib.pyplot as plt

from inference.train import train
from simulate.simulate import simulate
from plot.plot_hist_loss import plot_loss
from plot.plot_posterior import plot_posterior


FIG_FOLDER = 'fig/artificial'

os.makedirs(FIG_FOLDER, exist_ok=True)


SEED_DATA_GENERATION = 0


def plot_metric(hist_train, hist_val, path):

    metrics = list(hist_train.keys())

    n_row = len(metrics)

    fig, axes = plt.subplots(nrows=n_row, figsize=(10, 5))

    for i, k in enumerate(metrics):
        ax = axes[i]

        for hist, color in (hist_train, "C0"), (hist_val, "C1"):
            ax.plot(hist[k], color=color)

        ax.set_xlabel("epoch")
        ax.set_ylabel(k)

    fig.tight_layout()
    plt.savefig(path)


def make_fig(theta_flow, hist_train, hist_val, truth):

    print("Making the plots...")
    plot_posterior(
        path=f"{FIG_FOLDER}/posterior.pdf",
        theta_flow=theta_flow,
        truth=truth)
    plot_loss(
        path=f"{FIG_FOLDER}/loss.pdf",
        hist_loss=hist_train['free_energy'])

    plot_metric(
        path=f"{FIG_FOLDER}/hist_metric.pdf",
        hist_train=hist_train,
        hist_val=hist_val)


def main():

    dataset, truth = simulate(use_torch=True,
                              seed=SEED_DATA_GENERATION,
                              use_torch_dataset=True)

    z_flow, theta_flow, hist_train, hist_val, hist_comp_truth, config = train(
        dataset=dataset,
        batch_size=len(dataset),
        training_split=0.9,
        truth=truth,
        bkp_folder="bkp/run_oneshot_artificial",
        bkp_name="norm_flows",
        load_if_exists=False)

    make_fig(theta_flow=theta_flow,
             hist_train=hist_train,
             hist_val=hist_val,
             truth=truth)


if __name__ == "__main__":
    main()
