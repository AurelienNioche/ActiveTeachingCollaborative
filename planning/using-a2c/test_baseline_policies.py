import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from baseline_policies.threshold import Threshold
from baseline_policies.conservative import Conservative
from baseline_policies.leitner import Leitner

#  from environments.continuous_teaching import ContinuousTeaching
from environments.discontinuous_teaching import DiscontinuousTeaching

from environments.reward_types import types as reward_types

from human_agents import generate_agents


def create_env(seed=123, n_users=30, n_items=200,
               penalty_coeff=0, reward_type='mean_learned',
               tau=0.9, break_length=24 * 60 ** 2,
               time_per_iter=3,
               n_iter_session=100):

    forget_rates, repetition_rates = \
        generate_agents(n_users=n_users, n_items=n_items, seed=seed)

    env = DiscontinuousTeaching(
        tau=tau,
        n_item=n_items,
        break_length=break_length,
        time_per_iter=time_per_iter,
        n_iter_per_session=n_iter_session,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        delta_coeffs=np.array([3, 20]),
        n_coeffs=2,
        penalty_coeff=penalty_coeff,
        reward_type=reward_types[reward_type]
    )

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

    plot(actions=actions, rewards=rewards, env=env, title="threshold")


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

    plot(actions=actions, rewards=rewards, env=env, title="conservative")


def test_leitner():

    env = create_env()

    # Try Leitner ----------------------------------
    policy = Leitner(env=env, delay_factor=2, delay_min=3)
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

    plot(actions=actions, rewards=rewards, env=env, title="leitner")


def plot(actions, rewards, env, title=""):

    n_learned = np.array(rewards) * env.n_item

    fig, ax = plt.subplots()
    ax.plot(n_learned)
    ax.set_xlabel("time")
    ax.set_ylabel("n learned")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(actions)), actions, alpha=0.5)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def main():
    test_leitner()
    # test_threshold()
    # test_conservative()


if __name__ == "__main__":
    main()
