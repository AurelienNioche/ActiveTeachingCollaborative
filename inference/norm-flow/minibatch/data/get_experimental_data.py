import pandas as pd
import numpy as np
import torch


def get_experimental_data():
    df = pd.read_csv("../../data/data_full.csv", index_col=0)
    df.drop(df[(df.domain != "active.fi") | (df.n_session_done != 14)].index,
            inplace=True)
    df["ts_display"] = pd.to_datetime(df["ts_display"])  # str to datetime
    df["ts_reply"] = pd.to_datetime(df["ts_reply"])
    # Convert timestamps into seconds
    beginning_history = pd.Timestamp("1970-01-01", tz="UTC")
    df["timestamp"] = (
                df["ts_reply"] - beginning_history).dt.total_seconds().values
    # Copy actual item ID
    df["item_id"] = df.item
    for i, i_id in enumerate(df.item_id.unique()):
        df.loc[df.item_id == i_id, 'item'] = i

    n_u = len(df.user.unique())
    n_o_by_u = np.zeros(shape=n_u, dtype=int)
    for u, (user, user_df) in enumerate(df.groupby("user")):
        n_o_by_u[u] = len(user_df) - len(
            user_df.item.unique())  # Do not count first presentation

    n_obs = n_o_by_u.sum()

    y = np.zeros(shape=n_obs, dtype=int)
    x = np.zeros(shape=n_obs, dtype=float)
    w = np.zeros(shape=n_obs, dtype=int)
    r = np.zeros(shape=n_obs, dtype=int)
    u = np.zeros(shape=n_obs, dtype=int)

    idx = 0

    for i_u, (user, user_df) in enumerate(df.groupby("user")):

        user_df = user_df.sort_values(by="timestamp")
        seen = user_df.item.unique()
        w_u = user_df.item.values  # Words
        ts_u = user_df.timestamp.values
        counts = {word: -1 for word in seen}
        last_pres = {word: None for word in seen}
        r_u = np.zeros(len(user_df))  # Number of repetitions
        x_u = np.zeros(r_u.shape)  # Time elapsed since last repetition
        for i, word in enumerate(w_u):
            ts = ts_u[i]
            r_u[i] = counts[word]
            if last_pres[word] is not None:
                x_u[i] = ts - last_pres[word]
            counts[word] += 1
            last_pres[word] = ts

        to_keep = r_u >= 0
        y_u = user_df.success.values[to_keep]
        r_u = r_u[to_keep]
        w_u = w_u[to_keep]
        x_u = x_u[to_keep]

        n_ou = len(y_u)
        # assert n_o_by_u[i_u] == n_ou

        y[idx:idx + n_ou] = y_u
        x[idx:idx + n_ou] = x_u
        w[idx:idx + n_ou] = w_u
        r[idx:idx + n_ou] = r_u
        u[idx:idx + n_ou] = i_u

        idx += n_ou

    data = {
        'u': u, 'w': w,
        'x': torch.from_numpy(x.reshape(-1, 1)),
        'r': torch.from_numpy(r.astype(float).reshape(-1, 1)),
        'y': torch.from_numpy(y.astype(float).reshape(-1, 1))
    }

    n_w = len(np.unique(w))
    n_o_max = n_o_by_u.max()
    n_o_min = n_o_by_u.min()
    print("number of user", n_u)
    print("number of items", n_w)
    print("total number of observations (excluding first presentation)", n_obs)
    print("minimum number of observation for a single user", n_o_min)
    print("maximum number of observation for a single user", n_o_max)

    return data
