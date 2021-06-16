import logging
import sys

import numpy as np
import optuna

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching


def optimize_a2c(trial):
    """ Learning hyperparamters we want to optimise"""
    has_shared_net = trial.suggest_categorical('has_shared_net', [True, False])
    shared_layers_dim = []
    if has_shared_net:
        n_shared_layers = trial.suggest_int('n_shared_layers', 1, 3)
        for i in range(n_shared_layers):
            shared_layer_dim = trial.suggest_int(
                'shared_layer_dim{}'.format(i),
                4,
                64
            )
            shared_layers_dim += [shared_layer_dim]
    n_layers = trial.suggest_int('n_layers', 1, 4)
    layers_dim = []
    for i in range(n_layers):
        layer_dim = trial.suggest_int('layer_dim{}'.format(i), 4, 128)
        layers_dim += [layer_dim]

    return {
        # 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-7, 1.),
        # 'constant_lr': trial.suggest_categorical('constant_lr', [True, False]),
        # 'normalize_advantage': trial.suggest_categorical('normalize_advantage', [True, False]),
        "net_arch": shared_layers_dim + [{'pi': layers_dim, 'vf': layers_dim}]
    }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_a2c(trial)
    env = ContinuousTeaching(t_max=100, alpha=0.2, tau=0.9)
    model = A2C(
        env,
        learning_rate=5e-4,
        constant_lr=True,
        normalize_advantage=False,
        **model_params
    )

    iterations = env.t_max * 1000
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        model.learn(iterations, callback=callback)
        return -1 * np.mean(callback.hist_rewards[-100:])


if __name__ == '__main__':
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "planner-netarch-study"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=storage_name,
        load_if_exists=False
    )
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=4)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')