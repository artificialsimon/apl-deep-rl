import time
import gym
import gym_apl
#import numpy as np

from baselines import deepq


def callback(lcl, _glb):
    # stop training if reward exceeds 499
    #is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 1.1
    #print(sum(lcl['episode_rewards'][-101:-1]) / 100)
    #return is_solved
    return False


def main():
    ##np.seterr(all='raise')
    env = gym.make("apl-v0")
    #act = deepq.learn(
        #env,
        #network='mlp',
        #lr=1e-3,
        #checkpoint_freq=None,
        #total_timesteps=int(1e5),
        #buffer_size=50000,
        #exploration_fraction=0.1,
        #exploration_final_eps=0.02,
        #print_freq=10,
        #load_path="./models/apl-v0-dqn-20181003-151750",
        #callback=callback
    #)
    #timestr = time.strftime("%Y%m%d-%H%M%S")
    #act.save("./models/apl-v0-dqn-" + timestr)
    act = deepq.load_act("./models/apl-v0-dqn-20181003-152611.pickle")
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)

if __name__ == '__main__':
    main()
