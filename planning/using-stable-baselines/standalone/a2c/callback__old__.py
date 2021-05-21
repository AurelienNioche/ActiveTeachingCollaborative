from typing import Dict, Any

from tqdm.autonotebook import tqdm

import numpy as np


class ProgressBarCallback:
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, env, check_freq):
        self.pbar = None
        # The RL model
        self.model = None

        self.env = env
        self.check_freq = check_freq

        self.hist_rewards = []

    def init_callback(self, model: "a2c.A2C") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model

    def on_step(self):

        # Update the progress bar:
        self.pbar.n = self.model.num_timesteps
        # self.pbar.update()

        # Evaluate
        if self.model.num_timesteps % self.check_freq == 0:
            obs = self.env.reset()
            rewards = []
            while True:
                action = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)

                rewards.append(reward)

                if done:
                    break
            self.hist_rewards.append(rewards)
            self.pbar.set_postfix(
                {"average-reward": f"{np.mean(rewards):.4f}",
                 "max-reward": f"{np.max(rewards):.4f}"})

    def on_training_start(self, total_timesteps) -> None:
        # Those are reference and will be updated automatically
        self.pbar.reset(total=total_timesteps)

    def on_training_end(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        pass

    def __enter__(self):
        # create the progress bar and callback,
        # return the callback
        self.pbar = tqdm()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.model.num_timesteps
        self.pbar.close()
