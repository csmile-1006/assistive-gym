import argparse
import importlib

import gym
import numpy as np

# import assistive_gym


def sample_action(env, coop):
    if coop:
        return {"robot": env.action_space_robot.sample(), "human": env.action_space_human.sample()}
    return env.action_space.sample()


def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make("assistive_gym:" + env_name)
    else:
        module = importlib.import_module("assistive_gym.envs")
        env_class = getattr(module, env_name.split("-")[0] + "Env")
        env = env_class()
    env.seed(seed)
    return env


def viewer(env_name):
    coop = "Human" in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)
    i = 1

    while True:
        done = False
        env.render()
        observation = env.reset()
        action = sample_action(env, coop)
        if coop:
            print(
                "Robot observation size:",
                np.shape(observation["robot"]),
                "Human observation size:",
                np.shape(observation["human"]),
                "Robot action size:",
                np.shape(action["robot"]),
                "Human action size:",
                np.shape(action["human"]),
            )
        else:
            print("Observation size:", np.shape(observation), "Action size:", np.shape(action))

        while not done:
            observation, reward, done, info = env.step(sample_action(env, coop))
            if coop:
                done = done["__all__"]
            if i % 100 == 0:
                env.close()
                return
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assistive Gym Environment Viewer")
    parser.add_argument("--env", default="ScratchItchJaco-v1", help="Environment to test (default: ScratchItchJaco-v1)")
    args = parser.parse_args()

    viewer(args.env)
