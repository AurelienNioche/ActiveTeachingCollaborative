import numpy as np
import gym


class DummyVecEnv:
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env: a Gym environment
    """

    def __init__(self, env: gym.Env):
        self.envs = [env, ]
        self.num_envs = len(self.envs)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.buf_obs = np.zeros((self.num_envs, *env.observation_space.shape),
                                dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)

    def seed(self, seed: int) -> None:
        for i in range(self.num_envs):
            self.envs[i].seed(seed+i)

    def reset(self) -> np.ndarray:
        for i in range(self.num_envs):
            obs = self.envs[i].reset()
            self.buf_obs[i] = obs
        return np.copy(self.buf_obs)

    def close(self) -> None:
        for i in range(self.num_envs):
            self.envs[i].close()

    def step(self, actions: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """

        for i in range(self.num_envs):
            obs, self.buf_rews[i], self.buf_dones[i], _ = \
                self.envs[i].step(actions[i])
            if self.buf_dones[i]:
                obs = self.envs[i].reset()
            self.buf_obs[i] = obs
        return np.copy(self.buf_obs), np.copy(self.buf_rews), \
            np.copy(self.buf_dones), None
