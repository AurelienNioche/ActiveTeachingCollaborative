import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from a2c.a2c import A2C
from environments.continuous_teaching import ContinuousTeaching

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching, types
from environments.discontinuous_teaching import DiscontinuousTeaching
from human_agents import generate_agents

sns.set()
n_users = 5
n_items = 30


def produce_rates():
    global n_items, n_users
    forget_rates, repetition_rates = generate_agents(n_users, n_items, seed=123)
    print("forget", forget_rates.mean())
    print("repeat", repetition_rates.mean())
    return forget_rates, repetition_rates


def test_discontinuous_teaching():
    global n_items

    forget_rates, repetition_rates = produce_rates()
    env = DiscontinuousTeaching(
        tau=0.9,
        break_length=24 * 60 ** 2,
        time_per_iter=3,
        n_iter_per_session=100,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        delta_coeffs=np.array([3, 20]),
        n_coeffs=2,
        n_item=n_items,
        penalty_coeff=0.2
    )
    layers_dim = [64, 64, 128]
    m = A2C(env,
            net_arch=[{'pi': layers_dim, 'vf': layers_dim}],
            seed=123
        )

    env_t_max = env.n_session * env.n_iter_per_session
    iterations = env_t_max * 20000
    check_freq = env_t_max

    with ProgressBarCallback(env, check_freq) as callback:
        m.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.show()
    return m


def run_continuous_teaching(reward_type):
    global n_items

    forget_rates, repetition_rates = produce_rates()
    env = ContinuousTeaching(
        t_max=100,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        n_item=n_items,
        tau=0.9,
        delta_coeffs=np.array([3, 20]),
        n_coeffs=2,
        penalty_coeff=0.2,
        reward_coeff=20.,
        reward_type=reward_type
    )

    m = A2C(env, seed=123)

    iterations = env.t_max * 1e5
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        m.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.show()
    return m


if __name__ == "__main__":

    for x, y in types.items():
        print('Running on {}...'.format(x))
        model = run_continuous_teaching(y)
        model.env.all_forget_rates.tofile('continuous_runs/forget_{}'.format(x), sep=',', format='%s')
        model.env.all_repetition_rates.tofile('continuous_runs/repetition_{}'.format(x), sep=',', format='%s')
        model.save('continuous_runs/run_{}'.format(x))