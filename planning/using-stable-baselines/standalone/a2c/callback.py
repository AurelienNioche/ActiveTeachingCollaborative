import copy

from tqdm.autonotebook import tqdm

import numpy as np


class ProgressBarCallback:
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar, env, check_freq):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar
        self.env = env
        self.check_freq = check_freq

        self.hist_rewards = []

        self.n_calls = 0

        self.model = None

    # Type hint as string to avoid circular import
    def init_callback(self, model) -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model

    def on_training_start(self, tamere) -> None:
        pass

    def on_training_end(self) -> None:
        pass

    def on_rollout_start(self)-> None:
        pass

    def on_rollout_end(self) -> None:
        pass

    def on_step(self):
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        self.n_calls += 1
        # timesteps start at zero
        self.num_timesteps = self.model.num_timesteps

        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

        # Evaluate
        if self.n_calls % self.check_freq == 0:
            obs = self.env.reset()
            rewards = []
            while True:
                action = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)

                rewards.append(reward)

                if done:
                    break
            self.hist_rewards.append(rewards)
            self._pbar.set_postfix({"average-reward": f"{np.mean(rewards):.4}",
                                    "max-reward": f"{np.max(rewards):.4f}"})
        return True


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager:
    def __init__(self, total_timesteps, env,
                 check_freq):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.env = copy.deepcopy(env)
        self.check_freq = check_freq
        self.pbar_callback = None

    def __enter__(self):
        # create the progress bar and callback,
        # return the callback
        self.pbar = tqdm(total=self.total_timesteps)
        self.pbar_callback = ProgressBarCallback(self.pbar, self.env,
                                                 self.check_freq)
        return self.pbar_callback

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

    @property
    def hist_rewards(self):
        return self.pbar_callback.hist_rewards