import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from q2_schedule import LinearExploration, LinearSchedule
from q4_nature_torch import NatureQN

from configs.q6_train_atari_nature import Config

"""
Use deep Q network for the Atari game. Please report the final result.
Feel free to change the configurations (in the configs/ folder). 
If so, please report your hyper parameters.

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.

To launch tensorboard, open a Terminal window and run 
tensorboard --logdir=results/
Then, connect remotely to 
address-ip-of-the-server:6006 
6006 is the default port used by tensorboard.
"""
if __name__ == '__main__':
    # make env
    env = gym.make(Config.env_name)
    env = MaxAndSkipEnv(env, skip=Config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=Config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, Config.eps_begin,
                                     Config.eps_end, Config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(Config.lr_begin, Config.lr_end,
                                 Config.lr_nsteps)

    # train model
    model = NatureQN(env, Config)
    model.run(exp_schedule, lr_schedule)
