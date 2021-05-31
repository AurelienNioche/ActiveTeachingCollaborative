from inference.train import train
from inference.plannar import NormalizingFlow
from simulate.simulate import simulate
from plot.plot_hist_loss import plot_loss
from plot.plot_posterior import plot_posterior


def main():

    data, truth = simulate()
    z_flow, theta_flow, hist_loss = train(
        **data,
        freq_update_loss=1,
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
