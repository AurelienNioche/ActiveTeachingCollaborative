from inference.train import train
from inference.plot import plot_loss, plot_posterior
from simulate.simulate import simulate
from inference.plannar import NormalizingFlow


def main():

    data, truth = simulate()
    z_flow, theta_flow, hist_loss = train(
        **data,
        freq_update_loss=5,
        n_sample=40,
        epochs=5000)
    z_flow.save("z_flow_artificial")
    theta_flow.save("theta_flow_artificial")
    # z_flow = NormalizingFlow.load("z_flow_artificial")
    theta_flow = NormalizingFlow.load("theta_flow_artificial")
    plot_posterior(theta_flow=theta_flow,
                   truth=truth,
                   name="artificial")
    plot_loss(hist_loss=hist_loss, name="artificial")


if __name__ == "__main__":
    main()
