from typing import Any, Dict, Optional, Type, Union, Tuple, List, Callable
from functools import partial

import numpy as np

import torch
from torch import nn

import gym
from gym import spaces

from . nn import MlpExtractor
from . distribution import CategoricalDistribution

# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]


class FlattenExtractor(nn.Module):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__()

        self._observation_space = observation_space
        self.features_dim = spaces.utils.flatdim(observation_space)
        assert self.features_dim > 0

        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)


class ActorCriticPolicy(nn.Module):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr: float,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        # Action distribution
        assert isinstance(action_space, spaces.Discrete), \
            f"Error: probability distribution, " \
            f"not implemented for action space of type {type(action_space)}."
        self.action_dist = CategoricalDistribution(action_space.n)

        self.features_extractor = FlattenExtractor(self.observation_space)
        features_dim = self.features_extractor.features_dim

        self.mlp_extractor = MlpExtractor(
            features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:  # true by defaut
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        # Small values to avoid NaN in Adam optimizer
        if optimizer_class == torch.optim.Adam \
                and "eps" not in optimizer_kwargs:
            optimizer_kwargs["eps"] = 1e-5

        self.optimizer_kwargs = optimizer_kwargs  # for saving
        self.optimizer_class = optimizer_class    # for saving
        self.optimizer = optimizer_class(
            self.parameters(),
            lr=lr, **self.optimizer_kwargs)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_latent(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        features = self.features_extractor(obs.float())
        latent_pi, latent_vf = self.mlp_extractor(features)
        return latent_pi, latent_vf

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> CategoricalDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        assert isinstance(self.action_dist, CategoricalDistribution), "Invalid action distribution"
        # Here mean_actions are the logits before the softmax
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        assert not isinstance(observation, dict), "using ObsDictWrapper but no supported here"
        observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = torch.as_tensor(observation)
        with torch.no_grad():
            latent_pi, _ = self._get_latent(observation)
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions = distribution.get_actions(deterministic=deterministic)

        # Convert to numpy
        actions = actions.cpu().numpy()
        assert not isinstance(self.action_space, gym.spaces.Box), "should not be box"

        return actions

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """ (float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    def _get_constructor_parameters(self) -> Dict[str, Any]:

        return dict(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            lr_schedule=self._dummy_schedule,
            # dummy lr schedule, not needed for loading policy alone
            ortho_init=self.ortho_init,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            observation_space=self.observation_space,
            action_space=self.action_space)

    def save(self, path: str) -> None:
        """
        Save model to a given location.
        """
        torch.save({"state_dict": self.state_dict(),
                    "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path: str):
        """
        Load model from path.
        """
        saved_variables = torch.load(path)
        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        return model
