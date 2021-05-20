from collections import deque
from typing import Any, Dict, Optional, Type, Union, Tuple, List, Callable
import time

import numpy as np

import torch
from torch import nn
import gym
from gym import spaces
from torch.nn import functional as F

from . policy import ActorCriticPolicy
from . rollout import RolloutBuffer
from . environment_wrapper import Monitor
from . callback import BaseCallback, ProgressBarManager
from stable_baselines3.common.vec_env import DummyVecEnv

GymEnv = gym.Env
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]
TensorDict = Dict[str, torch.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[ProgressBarManager, BaseCallback]

# from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


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
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        env: GymEnv,
        learning_rate: float = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        normalize_advantage: bool = False,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
        policy_kwargs: Union[Dict, None] = None,
    ):
        # get VecNormalize object if needed
        self.verbose = verbose

        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0

        self.seed = seed

        self.start_time = None
        self.policy = None
        self.learning_rate = learning_rate

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

        # Wrap the env
        self.env = DummyVecEnv([lambda: Monitor(env)])

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.normalize_advantage = normalize_advantage

        self.lr_schedule = lambda _: self.learning_rate

        # Seed numpy RNG
        np.random.seed(seed)
        # seed torch RNG
        torch.manual_seed(seed)

        self.action_space.seed(seed)
        self.env.seed(seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda)

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if policy_kwargs is None:
            policy_kwargs = {}
        self.policy_kwargs = policy_kwargs
        if use_rms_prop:
            self.policy_kwargs["optimizer_class"] = torch.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99,
                                                          eps=rms_prop_eps,
                                                          weight_decay=0)

        self.policy = ActorCriticPolicy(
            self.observation_space,
            self.action_space,
            self.learning_rate,
            **self.policy_kwargs)

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

        self._n_updates += 1

    def collect_rollouts(self, callback: BaseCallback) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        self.rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < self.n_steps:

            with torch.no_grad():
                # Convert to pytorch tensor
                obs_tensor = torch.as_tensor(self._last_obs)
                actions, values, log_probs = self.policy.forward(obs_tensor.unsqueeze(0))
            actions = actions.cpu().numpy()

            new_obs, rewards, dones, _ = self.env.step(actions)

            self.num_timesteps += 1

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            n_steps += 1

            actions = actions.reshape(-1, 1)

            self.rollout_buffer.add(new_obs, actions,
                                    rewards,
                                    dones,
                                    values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with torch.no_grad():
            # Compute value for the last timestep
            obs_tensor = torch.as_tensor(new_obs)
            _, values, _ = self.policy.forward(obs_tensor)

        self.rollout_buffer.compute_returns_and_advantage(last_values=values,
                                                          dones=dones)

        callback.on_rollout_end()

        return True

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            reset_num_timesteps: bool = True):

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(callback)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps,
                                                    total_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []

    def _setup_learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            reset_num_timesteps: bool = True,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :return:
        """
        self.start_time = time.time()
        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

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
            self._last_dones = np.zeros(1, dtype=bool)  # n_env=1

        # Create eval callback if needed
        callback.init_callback(self)

        return total_timesteps, callback

    # def _update_info_buffer(self, infos: List[Dict[str, Any]],
    #                         dones: Optional[np.ndarray] = None) -> None:
    #     """
    #     Retrieve reward, episode length, episode success and update the buffer
    #     if using Monitor wrapper or a GoalEnv.
    #
    #     :param infos: List of additional information about the transition.
    #     :param dones: Termination signals
    #     """
    #     print("dones", dones)
    #     if dones is None:
    #         dones = np.array([False] * len(infos))
    #     for idx, info in enumerate(infos):
    #         maybe_ep_info = info.get("episode")
    #         maybe_is_success = info.get("is_success")
    #         if maybe_ep_info is not None:
    #             self.ep_info_buffer.extend([maybe_ep_info])
    #         if maybe_is_success is not None and dones[idx]:
    #             self.ep_success_buffer.append(maybe_is_success)

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
        learning_rate = self.lr_schedule(self._current_progress_remaining)

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

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
