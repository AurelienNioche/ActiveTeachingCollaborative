import numpy as np
import torch

from . generate_human_like_parameters import generate_learners_parameterization


class Leitner:

    def __init__(self, n_item,  delay_factor=2, delay_min=4):

        self.n_item = n_item

        self.delay_factor = delay_factor
        self.delay_min = delay_min

        self.box = np.full(self.n_item, -1)
        self.due = np.full(self.n_item, -1)

    def update_box_and_due_time(self, last_idx,
                                last_was_success, last_time_reply):

        if last_was_success:
            self.box[last_idx] += 1
        else:
            self.box[last_idx] = \
                max(0, self.box[last_idx] - 1)

        delay = self.delay_factor ** self.box[last_idx]
        # Delay is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ...
        self.due[last_idx] = \
            last_time_reply + self.delay_min * delay

    def _pickup_item(self, now):

        seen = np.argwhere(np.asarray(self.box) >= 0).flatten()
        n_seen = len(seen)

        if n_seen == self.n_item:
            return np.argmin(self.due)

        else:
            seen__due = np.asarray(self.due)[seen]
            seen__is_due = np.asarray(seen__due) <= now
            if np.sum(seen__is_due):
                seen_and_is_due__due = seen__due[seen__is_due]

                return seen[seen__is_due][np.argmin(seen_and_is_due__due)]
            else:
                return self._pickup_new()

    def _pickup_new(self):
        return np.argmin(self.box)

    def ask(self, now, last_was_success, last_time_reply, idx_last_q):

        if idx_last_q is None:
            item_idx = self._pickup_new()

        else:

            self.update_box_and_due_time(
                last_idx=idx_last_q,
                last_was_success=last_was_success,
                last_time_reply=last_time_reply)
            item_idx = self._pickup_item(now)

        return item_idx


def simulate_human_like_using_leitner(
        seed,
        file_population_param="data/param_exp_data.csv",
        n_u=53,
        n_w=1998,
        n_ss=6,
        n_iter_per_session=100,
        space_between_session=24 * 60 ** 2,
        time_per_iter=3):

    initial_forget_rates, repetition_effects, mu, sg_u, sg_w, Zu, Zw = \
        generate_learners_parameterization(
            seed=seed,
            n_users=n_u,
            n_items=n_w,
            file_population_param=file_population_param)

    break_length = space_between_session - time_per_iter * n_iter_per_session

    n_obs = n_iter_per_session * n_ss * n_u

    u = np.zeros(n_obs)
    w = np.zeros(n_obs)
    r = np.zeros(n_obs)
    x = np.zeros(n_obs)
    y = np.zeros(n_obs)

    i = 0

    for user in range(n_u):

        teacher = Leitner(n_item=n_w)

        rep = np.full(n_w, -1, dtype=int)
        last_pres = np.full(n_w, -np.inf)

        last_recall = None
        last_t = None
        last_item = None

        t = 0

        for session in range(n_ss):
            for iteration in range(n_iter_per_session):

                item = teacher.ask(
                    now=t,
                    last_was_success=last_recall,
                    last_time_reply=last_t,
                    idx_last_q=last_item)

                delta_last_pres = t - last_pres[item]
                rep_item = rep[item]

                a = initial_forget_rates[user, item]
                b = repetition_effects[user, item]

                neg_rate = - a * delta_last_pres * (1 - b) ** rep_item
                p = np.exp(neg_rate)
                recall = p > np.random.random()

                u[i] = user
                w[i] = item
                r[i] = rep_item
                x[i] = delta_last_pres
                y[i] = recall

                rep[item] += 1
                last_pres[item] = t

                i += 1

                last_recall = recall
                last_t = t
                last_item = item

                if iteration == n_iter_per_session - 1:
                    t += break_length
                else:
                    t += time_per_iter

    to_keep = r[:] >= 0

    u = u[to_keep]
    w = w[to_keep]
    r = r[to_keep]
    x = x[to_keep]
    y = y[to_keep]

    # Compute truth
    Z = mu + Zu[u.astype(int)] + Zw[w.astype(int)]

    sg_u_smp = np.std(Zu, axis=0)
    sg_w_smp = np.std(Zw, axis=0)
    mu_smp = np.mean(Z, axis=0)
    truth = {'mu': mu, 'sg_u': sg_u, 'sg_w': sg_w,
             'mu_smp': mu_smp, 'sg_u_smp': sg_u_smp, 'sg_w_smp': sg_w_smp}

    data = {
        'x': torch.from_numpy(x.reshape(-1, 1)),
        'y': torch.from_numpy(y.reshape(-1, 1)),
        'r': torch.from_numpy(r.reshape(-1, 1)),
        'u': u,
        'w': w}

    n_w = len(np.unique(w))
    print("Number of user", n_u)
    print("Number of items", n_w)
    print("Total number of observations (excluding first presentation)", n_obs)
    print("Number of observation for a single user", n_ss*n_iter_per_session)

    return data, truth
