import numpy as np
import pandas as pd
import scipy.special


def generate_learners_parameterization(n_users, n_items, seed,
                                       file_population_param):
    np.random.seed(seed)
    df_param = pd.read_csv(file_population_param, index_col=0)
    mu = np.array([df_param.loc["unconstrained", f"mu{i}"]
                   for i in (1, 2)])
    sig_users = np.array([df_param.loc["unconstrained", f"sigma_u{i}"]
                          for i in (1, 2)])
    sig_items = np.array([df_param.loc["unconstrained", f"sigma_w{i}"]
                          for i in (1, 2)])

    z_user = np.random.normal(np.zeros(2), sig_users, size=(n_users, 2))
    z_item = np.random.normal(np.zeros(2), sig_items, size=(n_items, 2))
    initial_forget_rates = np.zeros((n_users, n_items))
    repetition_effects = np.zeros((n_users, n_items))
    for i in range(n_users):
        initial_forget_rates[i] = mu[0] + z_user[i, 0] + z_item[:, 0]
        repetition_effects[i] = mu[1] + z_user[i, 1] + z_item[:, 1]

    initial_forget_rates = np.exp(initial_forget_rates)
    repetition_effects = scipy.special.expit(repetition_effects)

    return initial_forget_rates, repetition_effects, \
        mu, sig_users, sig_items, z_user, z_item
