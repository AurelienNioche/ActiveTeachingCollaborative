import os
import matplotlib.pyplot as plt

from inference.train import train
from simulate.simulate import simulate
from plot.plot_hist_loss import plot_loss
from plot.plot_posterior import plot_posterior


SEED_DATA_GENERATION = 0


def plot_metric(hist_train, hist_val, path):

    metrics = list(hist_train.keys())

    n_row = len(metrics)

    fig, axes = plt.subplots(nrows=n_row, figsize=(10, n_row*5))

    for i, k in enumerate(metrics):
        ax = axes[i]

        for hist, color in (hist_train, "C0"), (hist_val, "C1"):
            ax.plot(hist[k], color=color)

        ax.set_xlabel("epoch")
        ax.set_ylabel(k)

    fig.tight_layout()
    plt.savefig(path)


def make_fig(theta_flow, hist, truth, fig_folder):

    os.makedirs(fig_folder, exist_ok=True)

    print("Making the plots...")
    plot_posterior(
        path=f"{fig_folder}/posterior.pdf",
        theta_flow=theta_flow,
        truth=truth)

    plot_loss(
        path=f"{fig_folder}/loss.pdf",
        hist_loss=hist['train']['free_energy'])

    plot_metric(
        path=f"{fig_folder}/hist_metric.pdf",
        hist_train=hist['train'],
        hist_val=hist['val'])

    plot_metric(
        path=f"{fig_folder}/hist_comp_truth.pdf",
        hist_train=hist['comp_truth_train'],
        hist_val=hist['comp_truth_val'])


def main():

    dataset, truth = simulate(use_torch=True,
                              seed=SEED_DATA_GENERATION,
                              use_torch_dataset=True)

    # bkp_name = "run_oneshot_artificial"

    # z_flow, theta_flow, hist, config = train(
    #     epochs=5000,
    #     initial_lr=0.01,
    #     dataset=dataset,
    #     batch_size=len(dataset),
    #     training_split=1.0,
    #     truth=truth,
    #     bkp_folder="bkp",
    #     bkp_name=bkp_name,
    #     load_if_exists=False)
    #
    # make_fig(
    #     fig_folder=f"fig/{bkp_name}",
    #     theta_flow=theta_flow,
    #     hist=hist,
    #     truth=truth)

    for batch_size, learning_rate in (len(dataset), 0.01), (64, 10e-8), (int(0.10*len(dataset)), 10e-8):
        for training_split in (0.9, ):

            bkp_name = f"run_oneshot_artificial_bs{batch_size}_ts{int(training_split*100)}"

            print(f"run {bkp_name}")

            z_flow, theta_flow, hist, config = train(
                epochs=5000,
                initial_lr=learning_rate,
                dataset=dataset,
                batch_size=batch_size, #len(dataset),
                training_split=training_split,
                truth=truth,
                bkp_folder="bkp",
                bkp_name=bkp_name,
                load_if_exists=False)

            make_fig(
                fig_folder=f"fig/{bkp_name}",
                theta_flow=theta_flow,
                hist=hist,
                truth=truth)


if __name__ == "__main__":
    main()
