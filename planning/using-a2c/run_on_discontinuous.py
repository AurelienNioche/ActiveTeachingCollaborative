import random

import matplotlib.pyplot as plt
import pandas as pd
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
n_items = 60


def produce_rates():
    global n_items, n_users
    forget_rates, repetition_rates = generate_agents(n_users, n_items)
    print("forget", forget_rates.mean())
    print("repeat", repetition_rates.mean())
    return forget_rates, repetition_rates


def run_discontinuous_teaching(reward_type, forgets, repetitions, gamma):
    global n_items

    # forget_rates, repetition_rates = produce_rates()
    env = DiscontinuousTeaching(
        tau=0.9,
        break_length=24 * 60 ** 2,
        time_per_iter=3,
        n_iter_per_session=100,
        initial_forget_rates=forgets,
        initial_repetition_rates=repetitions,
        delta_coeffs=np.array([3, 20]),
        n_item=n_items,
        penalty_coeff=0.2,
        reward_type=reward_type,
        gamma=gamma
    )
    # layers_dim = [64, 64, 128]
    m = A2C(env,
            # net_arch=[{'pi': layers_dim, 'vf': layers_dim}],
            # seed=123
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
    global n_items, rc

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
        reward_coeff=rc,
        reward_type=reward_type
    )

    m = A2C(env)

    iterations = env.t_max * 1e5
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        m.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.show()
    return m


if __name__ == "__main__":
    # for rc in [1, 1.5, 2, 3, 4]:
    #     print('Running on {}...'.format(rc))
    for i in [0.5, 1, 2, 3, 4]:
        print('Running on {}...'.format(i))
        forgets = pd.read_csv('data/forget_2', delimiter=',', header=None)
        repetitions = pd.read_csv('data/repetition_2', delimiter=',', header=None)
        forgets = np.array(forgets)[0]
        forgets = np.reshape(forgets, newshape=(n_users, n_items))
        repetitions = np.array(repetitions)[0]
        repetitions = np.reshape(repetitions, newshape=(n_users, n_items))
        model = run_discontinuous_teaching(types['exam_based'], forgets, repetitions, i)
        model.save('discontinuous_runs/eb_run_{}'.format(i))

    # model.env.all_forget_rates.tofile('discontinuous_runs/forget_{}'.format(rc), sep=',', format='%s')
    # model.env.all_repetition_rates.tofile('discontinuous_runs/repetition_{}'.format(rc), sep=',', format='%s')

    # for r, i in types.items():
    #     if i > 2:
    #         print('Running on {}...'.format(r))
    #         model = run_discontinuous_teaching(types[r])
    #         model.env.all_forget_rates.tofile('discontinuous_runs/forget_{}'.format(r), sep=',', format='%s')
    #         model.env.all_repetition_rates.tofile('discontinuous_runs/repetition_{}'.format(r), sep=',', format='%s')
    #         model.save('discontinuous_runs/run_{}'.format(r))
