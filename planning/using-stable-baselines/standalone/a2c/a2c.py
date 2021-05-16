from collections import deque
from typing import Any, Dict, Optional, Type, Union, Tuple, List
import time

import numpy as np

import torch
from torch import nn
import gym
from gym import spaces
from torch.nn import functional as F

from . policy import ActorCriticPolicy
from . rollout import RolloutBuffer

# from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
# from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
# from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common import logger, utils
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, EvalCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.monitor import Monitor


def safe_mean(arr: Union[np.ndarray, list, deque]) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr:
    :return:
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def maybe_make_env(env: Union[GymEnv, str, None], verbose: int) -> Optional[GymEnv]:
    """If env is a string, make the environment; otherwise, return env.

    :param env: The environment to learn from.
    :param verbose: logging verbosity
    :return A Gym (vector) environment.
    """
    if isinstance(env, str):
        if verbose >= 1:
            print(f"Creating environment from the given name '{env}'")
        env = gym.make(env)
    return env


def unwrap_vec_wrapper(env: Union["GymEnv", VecEnv], vec_wrapper_class: Type[VecEnvWrapper]) -> Optional[VecEnvWrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env:
    :param vec_wrapper_class:
    :return:
    """
    env_tmp = env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, vec_wrapper_class):
            return env_tmp
        env_tmp = env_tmp.venv
    return None


def unwrap_vec_normalize(env: Union["GymEnv", VecEnv]) -> Optional[VecNormalize]:
    """
    :param env:
    :return:
    """
    return unwrap_vec_wrapper(env, VecNormalize)  # pytype:disable=bad-return-type


class A2C:
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param normalize_advantage: Whether to normalize or not the advantage
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
        supported_action_spaces = (
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        monitor_wrapper: bool = True,
        support_multi_env=True,
    ):

        self.device = torch.device("cpu")
        if verbose > 0:
            print(f"Using {self.device} device")

        self.env = None  # type: Optional[GymEnv]
        # get VecNormalize object if needed
        self._vec_normalize_env = unwrap_vec_normalize(env)
        self.verbose = verbose

        self.observation_space = None  # type: Optional[gym.spaces.Space]
        self.action_space = None  # type: Optional[gym.spaces.Space]
        self.n_envs = None
        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        self.eval_env = None
        self.seed = seed
        self.action_noise = None  # type: Optional[ActionNoise]
        self.start_time = None
        self.policy = None
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.lr_schedule = None  # type: Optional[Schedule]
        self._last_obs = None  # type: Optional[np.ndarray]
        self._last_dones = None  # type: Optional[np.ndarray]
        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[np.ndarray]
        self._episode_num = 0
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1
        # Buffers for logging
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int

        # Create and wrap the env if needed
        if env is not None:
            if isinstance(env, str):
                if create_eval_env:
                    self.eval_env = maybe_make_env(env, self.verbose)

            env = maybe_make_env(env, self.verbose)
            env = self._wrap_env(env, self.verbose, monitor_wrapper)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            if supported_action_spaces is not None:
                assert isinstance(self.action_space, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} as action spaces "
                    f"but {self.action_space} was provided"
                )

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        self.normalize_advantage = normalize_advantage

        self.lr_schedule = lambda _: self.learning_rate

        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(seed)

        self.action_space.seed(seed)
        self.env.seed(seed)
        if self.eval_env is not None:  # None here!
            self.eval_env.seed(seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        self.policy_kwargs = {}
        if use_rms_prop:
            self.policy_kwargs["optimizer_class"] = torch.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99,
                                                          eps=rms_prop_eps, weight_decay=0)

        self.policy = ActorCriticPolicy(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # TODO: avoid second computation of everything because of the gradient
            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/explained_variance", explained_var)
        logger.record("train/entropy_loss", entropy_loss.item())
        logger.record("train/policy_loss", policy_loss.item())
        logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

    def collect_rollouts(
            self, env: VecEnv, callback: BaseCallback,
            rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:

            with torch.no_grad():
                # Convert to pytorch tensor
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low,
                                          self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones,
                               values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with torch.no_grad():
            # Compute value for the last timestep
            obs_tensor = torch.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values,
                                                     dones=dones)

        callback.on_rollout_end()

        return True

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes,
            eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback,
                                                      self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps,
                                                    total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(
                        self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean(
                        [ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean(
                        [ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed",
                              int(time.time() - self.start_time),
                              exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps,
                              exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self


    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose:
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])

        if (
            is_image_space(env.observation_space)
            and not is_vecenv_wrapped(env, VecTransposeImage)
            and not is_image_space_channels_first(env.observation_space)
        ):
            if verbose >= 1:
                print("Wrapping the env in a VecTransposeImage.")
            env = VecTransposeImage(env)

        # check if wrapper for dict support is needed when using HER
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            env = ObsDictWrapper(env)

        return env

    def _setup_learn(
            self,
            total_timesteps: int,
            eval_env: Optional[GymEnv],
            callback: MaybeCallback = None,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()
        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            print("reset action noise")
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()
            self._last_dones = np.zeros((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs
        utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name,
                               reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, eval_env, eval_freq,
                                       n_eval_episodes, log_path)

        return total_timesteps, callback

    def _get_eval_env(self, eval_env: Optional[GymEnv]) -> Optional[GymEnv]:
        """
        Return the environment that will be used for evaluation.

        :param eval_env:)
        :return:
        """
        if eval_env is None:
            eval_env = self.eval_env

        if eval_env is not None:
            eval_env = self._wrap_env(eval_env, self.verbose)
            assert eval_env.num_envs == 1
        return eval_env

    def _init_callback(
        self,
        callback: MaybeCallback,
        eval_env: Optional[VecEnv] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations; if None, do not evaluate.
        :param n_eval_episodes: How many episodes to play per evaluation
        :param n_eval_episodes: Number of episodes to rollout during evaluation.
        :param log_path: Path to a folder where the evaluations will be saved
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Create eval callback in charge of the evaluation
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_path,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
            )
            callback = CallbackList([callback, eval_callback])

        callback.init_callback(self)
        return callback

    def get_env(self) -> Optional[VecEnv]:
        """
        Returns the current environment (can be None if not defined).

        :return: The current environment
        """
        return self.env

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """

        def update_learning_rate(optimizer: torch.optim.Optimizer,
                                 learning_rate: float) -> None:
            """
            Update the learning rate for a given optimizer.
            Useful when doing linear schedule.

            :param optimizer:
            :param learning_rate:
            """
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        # Log the current learning rate
        logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: the input observation
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, deterministic)
