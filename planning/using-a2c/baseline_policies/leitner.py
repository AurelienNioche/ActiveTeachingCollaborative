import numpy as np


class Leitner:

    def __init__(self, env,  delay_factor=2, delay_min=4):

        self.n_item = env.n_item

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

    def act(self, obs):

        """
        Adaptation for RL env
        """

        # TODO: Adapt the code
