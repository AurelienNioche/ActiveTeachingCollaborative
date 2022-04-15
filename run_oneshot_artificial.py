import os

from inference.train import train
from simulate.simulate import simulate
from plot.plot_hist_loss import plot_loss
from plot.plot_posterior import plot_posterior


FIG_FOLDER = 'fig/artificial'

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


def main():

    dataset, truth = simulate(use_torch=True, seed=SEED_DATA_GENERATION,
                              use_torch_dataset=True)

    z_flow, theta_flow, hist_loss, hist_val, hist_train = train(
        dataset=dataset,
        batch_size=len(dataset),
        training_split=1.0,
        truth=truth,
        bkp_folder="bkp/run_oneshot_artificial",
        bkp_name="norm_flows",
        load_if_exists=True)
    make_fig(theta_flow=theta_flow, hist_loss=hist_loss, truth=truth)


if __name__ == "__main__":
    main()
