import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from baseline_policies.threshold import Threshold
from baseline_policies.conservative import Conservative

#  from environments.continuous_teaching import ContinuousTeaching
from environments.discontinuous_teaching import DiscontinuousTeaching

from human_agents import generate_agents


def create_env():
    n_users = 30
    n_items = 200

    np.random.seed(123)

    forget_rates, repetition_rates = generate_agents(n_users, n_items)

    env = DiscontinuousTeaching(
        tau=0.9,
        n_item=n_items,
        break_length=24 * 60 ** 2,
        time_per_iter=3,
        n_iter_per_session=100,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        delta_coeffs=np.array([3, 20]),
        n_coeffs=2,
        penalty_coeff=0.3
    )
    env.penalty_coeff = 0

    return env


def test_threshold():

    env = create_env()

    # Try Myopic ----------------------------------
    policy = Threshold(env=env)
    rewards = []
    actions = []

    obs = env.reset()
    while True:
        action = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break

    plot(actions=actions, rewards=rewards, env=env)


def test_conservative():

    env = create_env()

    # Try Conservative ----------------------------------
    policy = Conservative(env=env)
    rewards = []
    actions = []

    obs = env.reset()

    with tqdm(total=env.n_iter_per_session*env.n_session) as pb:
        while True:
            action = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)

            pb.update()

            if done:
                break

    plot(actions=actions, rewards=rewards, env=env)


def plot(actions, rewards, env):

    n_learned = np.array(rewards) * env.n_item

    fig, ax = plt.subplots()
    ax.plot(n_learned)
    ax.set_xlabel("time")
    ax.set_ylabel("n learned")
    plt.show()

    plt.scatter(np.arange(len(actions)), actions, alpha=0.5)
    plt.show()


def main():
    test_threshold()
    test_conservative()


if __name__ == "__main__":
    main()
