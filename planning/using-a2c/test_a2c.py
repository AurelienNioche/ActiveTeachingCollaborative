import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from a2c.a2c import A2C
from environments.continuous_teaching import ContinuousTeaching

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching
from human_agents import generate_agents

sns.set()


def test_save_and_load():

    model = A2C(env=ContinuousTeaching())
    path = "bkp/a2c_test.p"
    model.save(path)
    model.load(path)


def test_continuous_teaching():

    n_users = 100
    n_items = 200
    user = random.randint(0, n_users)
    forget_rates, repetition_rates = generate_agents(n_users, n_items)
    print("forget", forget_rates.mean())
    print("repeat", repetition_rates.mean())

    env = ContinuousTeaching(
        t_max=100,
        initial_forget_rates=forget_rates[user],
        initial_repetition_rates=repetition_rates[user],
        n_item=n_items,
        tau=0.9)

    model = A2C(env, seed=123)

    iterations = env.t_max * 1000
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        model.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.show()


if __name__ == "__main__":

    test_continuous_teaching()
