"""Support methods and classes for mario game environement."""
import gym
from gym.spaces import Box
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch
from torchvision import transforms


# @title Support method to create env and downsample
# Reference: https://github.com/pytorch/tutorials/blob/master/intermediate_source/mario_rl_tutorial.py
# -------------------Start of Support method-------------------
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = transforms.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        my_transforms = transforms.Compose(
            [transforms.Resize(self.shape), transforms.Normalize(0, 255)]
        )
        observation = my_transforms(observation).squeeze(0)
        return observation
# -------------------End of Support method-------------------


def build_single_env(mario_game, action_list, is_human_view=False):
    """Builds a single env for a game."""
    env = gym_super_mario_bros.make(mario_game)
    if is_human_view:
        env = JoypadSpace(env, action_list)
        return env
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = JoypadSpace(env, action_list)
    return env


def build_rotation_envs(mario_game_list, action_list, is_human_view=False):
    """Builds a list of envs for each game in the list."""
    return [build_single_env(k, action_list, is_human_view) for k in mario_game_list]


class ShuffleEnv(gym.Env):
    """A custom game evn that will reset game stage env evry time."""
    def __init__(self, mario_game_list, action_list, is_human_view):
        super(ShuffleEnv, self).__init__()
        self.envs = build_rotation_envs(mario_game_list, action_list, is_human_view)
        self.game_list = mario_game_list
        self.cur = -1

    def _next_env(self):
        self.name = self.game_list[self.cur]
        self.env = self.envs[self.cur]
        self.action_space = self.env.action_space
        self.render = self.env.render

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.cur = (self.cur + 1) % len(self.game_list)
        self._next_env()
        self.env.reset()

    def _blur(self, obs):
        return obs
