import time
import gym
import gym.spaces
import gym_apl
import random
from gym_apl.envs import apl_env
from baselines.ppo2 import ppo2
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines import bench, logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
import numpy as np
import os


def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(
                logger.get_dir(), str(rank)), allow_early_resets=True)
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def main():
    n_envs = 1
    env = make_custom_env('apl-v0', n_envs, 10)
    n_k = {'num_layers': 4, 'num_hidden': 128}
    #model = ppo2.learn(network='mlp', env=env, total_timesteps=(1e5)*5)
    model = ppo2.learn(network='mlp', env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        load_path='./models/apl-v0-1-20181003-170601.mlp',
        total_timesteps=n_envs * 6e5)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model.save("./models/apl-v0-1-" + timestr + ".mlp")
    #model.load('./models/apl-v0-1-20180904-132545.mlp')


    env = make_custom_env('apl-v0', n_envs, 10)
    obs = env.reset()
    while True:
        actions = model.step(obs)
        obs, _, done, _ = env.step(actions[0])
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            obs = env.reset()

    #env = gym.make('apl-v0')
    #obs = env.reset()
    #while True:
        #actions = model.step(obs)[0]
        #obs, _, done, _ = env.step(actions)
        #env.render()
        #done = done.any() if isinstance(done, np.ndarray) else done
        #if done:
            #obs = env.reset()
    #return 1


if __name__ == '__main__':
    main()
