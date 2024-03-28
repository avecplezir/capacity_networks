import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from typing import Union
import numpy as np
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Box


def make_env(env_id, seed, idx, capture_video, run_name):
    if 'MinAtar' in env_id:
        print('making MinAtar environment')
        make_env = make_env_minatary(env_id, seed, idx, capture_video, run_name)
    elif 'LunarLander' in env_id:
        print('making LunarLander rgb environment')
        make_env = make_env_luna(env_id, seed, idx, capture_video, run_name)
    else:
        print('making Atari environment')
        make_env = make_env_atati(env_id, seed, idx, capture_video, run_name)
    return make_env


def make_env_atati(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

def make_env_minatary(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(seed)

        return env

    return thunk


from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
def make_env_luna(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.PixelObservationWrapper(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = ResizeObservationLunar(env, (84, 84))
        # env = gym.wrappers.TransformObservation(env, lambda obs: obs.transpose(2, 0, 1))
        # env = VecTransposeImage(env)

        env.action_space.seed(seed)

        return env

    return thunk

class ResizeObservationLunar(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape: Union[tuple, int]):
        """Resizes image observations to shape given by :attr:`shape`.

        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        super().__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape

        self.shape = tuple(shape)
        obs_shape = self.shape + env.observation_space['pixels'].shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """
        try:
            import cv2
        except ImportError:
            raise DependencyNotInstalled(
                "opencv is not install, run `pip install gym[other]`"
            )

        observation = cv2.resize(
            observation['pixels'], self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation