from inference.flows import NormalizingFlow
from simulate.simulate import simulate
from inference.train_minibatch import train_minibatch

from plot.plot_hist_loss import plot_loss
from plot.plot_posterior import plot_posterior


def main():

    run_name = "artificial_minibatch"

    data, truth = simulate(use_torch=True)
    z_flow, theta_flow, hist_loss = train_minibatch(
        data,
        batch_size=1000,
        optimizer_name='RMSprop',
        constant_lr=False,
        n_sample=73,
        initial_lr=0.0017627035177522651,
        flow_length=39,
        n_epochs=5000)

    z_flow.save(f"z_flow_{run_name}")
    theta_flow.save(f"theta_flow_{run_name}")
    # z_flow = NormalizingFlow.load(f"z_flow_{run_name}")
    theta_flow = NormalizingFlow.load(f"theta_flow_{run_name}")
    plot_posterior(theta_flow=theta_flow,
                   truth=truth,
                   name=run_name)
    plot_loss(hist_loss=hist_loss, name=run_name)


if __name__ == "__main__":
    main()
