import logging
import sys
import random

import numpy as np
import optuna

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching

from human_agents import generate_agents

n_users = 100
n_items = 30
random.seed(123)
user = random.randint(0, n_users)
forget_rates, repetition_rates = generate_agents(n_users, n_items)
forget_rates.tofile('forget_rates.csv', sep=',')
repetition_rates.tofile('repetition_rates.csv', sep=',')


def run_one_episode(env, policy):
    rewards = []
    actions = []
    env.penalty_coeff=0.0
    obs = env.reset()
    while True:
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        actions.append(action)

        if done:
            obs = env.reset()
            break

    return rewards, actions


def optimize_coeff(trial):
    """ Learning hyper-parameters we want to optimise"""
    coeff = trial.suggest_float('n_coeffs', 0.0, 1.0)
    return {
        'penalty_coeff': coeff,
    }


def optimize_agent(trial, ):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    global forget_rates, repetition_rates, user
    env_params = optimize_coeff(trial)

    env = ContinuousTeaching(
        t_max=100,
        tau=0.9,
        initial_forget_rates=forget_rates[user],
        initial_repetition_rates=repetition_rates[user],
        n_coeffs=2,
        delta_coeffs=np.array([40, 7]),
        **env_params
    )

    model = A2C(
        env,
        learning_rate=5e-4,
        constant_lr=True,
        normalize_advantage=False,
    )

    iterations = env.t_max * 1000
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        model.learn(iterations, callback=callback)
    rewards, actions = run_one_episode(env, model)
    return np.sum(rewards)


if __name__ == '__main__':

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "penalty_coeff_study"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )
    try:
        study.optimize(optimize_agent, n_trials=200, n_jobs=4)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
