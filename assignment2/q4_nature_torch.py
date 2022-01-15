import torch.nn as nn

from utils.test_env import EnvTest
from q2_schedule import LinearExploration, LinearSchedule
from q3_linear_torch import Linear

from configs.q4_nature import config


class ModelDQN(nn.Module):
    def __init__(self, input_size, state_history, n_channels, num_actions):
        super(ModelDQN, self).__init__()
        modules = []
        pad = self.padding_size(input_size, 8, 4)
        modules.append(nn.Conv2d(n_channels * state_history, 32, kernel_size=(8, 8), stride=(4, 4), padding=pad))
        modules.append(nn.ReLU())
        out_size = self.output_size(input_size, kernel_size=8, stride=4, pad=pad, dilation=1)

        pad = self.padding_size(input_size, 4, 2)
        modules.append(nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=pad))
        modules.append(nn.ReLU())
        out_size = self.output_size(out_size, kernel_size=4, stride=2, pad=pad, dilation=1)

        pad = self.padding_size(input_size, 3, 1)
        modules.append(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=pad))
        modules.append(nn.ReLU())
        out_size = self.output_size(out_size, kernel_size=3, stride=1, pad=1, dilation=1)

        modules.append(nn.Flatten())
        modules.append(nn.Linear(out_size * out_size * 64, 512))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(512, num_actions))
        self.network = nn.ModuleList(modules)

    @staticmethod
    def padding_size(input_size, kernel_size, stride):
        return ((stride - 1) * input_size - stride + kernel_size) // 2

    @staticmethod
    def output_size(input_size, kernel_size=1, stride=1, pad=0, dilation=1):
        return ((input_size + (2 * pad) - (dilation * (kernel_size - 1)) - 1) // stride) + 1

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def initialize_models(self):
        """
        Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            (stride - 1) * img_height - stride + filter_size) // 2

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        ##############################################################
        self.q_network = ModelDQN(img_height, self.config.state_history, n_channels, num_actions)
        self.target_network = ModelDQN(img_height, self.config.state_history, n_channels, num_actions)
        ##############################################################

    def get_q_values(self, state, network='q_network'):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """

        ##############################################################
        state = state.permute(0, 3, 1, 2)
        out = self.q_network(state) if network == 'q_network' else self.target_network(state)
        ##############################################################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
                                     config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
