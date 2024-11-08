import argparse
import sys

import gym
import numpy as np

import assistive_gym  # noqa

parser = argparse.ArgumentParser(description="Assistive Gym Environment Viewer")
parser.add_argument("--env", default="DressingBaxter-v0", help="Environment to test (default: ScratchItchJaco-v0)")
args = parser.parse_args()

env = gym.make(args.env)

images = []
view = "front"
i = 0

while True:
    done = False
    env.render()
    observation = env.reset()
    action = env.action_space.sample()
    print("Observation size:", np.shape(observation), "Action size:", np.shape(action))
    while not done:
        # env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        images.append(env.get_camera_image_depth(view=view)[0])
        if i > 10:
            break
        i += 1
    break

from numpngw import write_apng

write_apng("animation.png", images, delay=100)
