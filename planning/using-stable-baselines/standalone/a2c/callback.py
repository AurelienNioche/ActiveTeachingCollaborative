from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import copy

from tqdm.autonotebook import tqdm

import numpy as np


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose:
    """

    def __init__(self, verbose: int = 0):
        super(BaseCallback, self).__init__()
        # The RL model
        self.model = None
        # An alias for self.model.get_env(), the environment used for training
        self.training_env = None
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.num_timesteps = 0  # type: int
        self.verbose = verbose
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}
        self.logger = None
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    # Type hint as string to avoid circular import
    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self.training_env = model.get_env()
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        # timesteps start at zero
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        :param locals_: the local variables during rollout collection
        """
        pass


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar, env, check_freq):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar
        self.env = env
        self.check_freq = check_freq

        self.hist_rewards = []

    def _on_step(self):
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
            self._pbar.set_postfix({"average-reward": np.mean(rewards),
                                    "max-reward": np.max(rewards)})


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager:
    def __init__(self, total_timesteps, env,
                 check_freq):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.env = copy.deepcopy(env)
        self.check_freq = check_freq
        self.pbar_callback = None

    def __enter__(
            self):  # create the progress bar and callback, return the callback
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
        print("hey")
        return self.pbar_callback.hist_rewards