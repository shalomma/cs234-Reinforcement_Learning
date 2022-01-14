import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from utils.test_env import EnvTest
from core.deep_q_learning_torch import DQN
from q2_schedule import LinearExploration, LinearSchedule

from configs.q3_linear import Config


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """

    def initialize_models(self):
        """
        Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a linear layer with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. Look up nn.Linear
        """
        # this information might be useful
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        ##############################################################

        self.q_network = \
            torch.nn.Sequential(nn.Linear(img_height * img_width * n_channels * self.config.state_history, num_actions))
        self.target_network = \
            torch.nn.Sequential(nn.Linear(img_height * img_width * n_channels * self.config.state_history, num_actions))

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
            1. Look up torch.flatten
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ##############################################################

        assert network in ['q_network', 'target_network'], 'unknown network input'
        inputs = torch.flatten(state, start_dim=1)
        out = self.q_network(inputs) if network == 'q_network' else self.target_network(inputs)

        ##############################################################

        return out

    def update_target(self):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights.

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.

        Hint:
            1. look up saving and loading pytorch models
        """

        ##############################################################

        self.target_network.load_state_dict(self.q_network.state_dict())

        ##############################################################

    def calc_loss(self, q_values: Tensor, target_q_values: Tensor,
                  actions: Tensor, rewards: Tensor, done_mask: Tensor) -> Tensor:
        """
        Calculate the MSE loss of this step.
        The loss for an example is defined as:
            Q_samp(s) = r if done
                        = r + gamma * max_a' Q_target(s', a')
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')
            target_q_values: (torch tensor) shape = (batch_size, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')

            : (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
            rewards: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
            done_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state

        Hint:
            You may find the following functions useful
                - torch.max
                - torch.sum
                - torch.nn.functional.one_hot
                - torch.nn.functional.mse_loss
        """
        # you may need this variable
        num_actions = self.env.action_space.n
        gamma = self.config.gamma

        ##############################################################
        Q_samp = rewards + gamma * target_q_values.max(dim=1).values
        Q_samp[done_mask] = rewards[done_mask]
        Q = torch.sum(q_values * F.one_hot(actions.to(torch.int64), num_classes=num_actions), dim=1)
        return F.mse_loss(Q, Q_samp)
        ##############################################################

    def add_optimizer(self):
        """
        Set self.optimizer to be an Adam optimizer optimizing only the self.q_network
        parameters

        Hint:
            - Look up torch.optim.Adam
            - What are the input to the optimizer's constructor?
        """
        ##############################################################
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        ##############################################################


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, Config.eps_begin,
                                     Config.eps_end, Config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(Config.lr_begin, Config.lr_end,
                                 Config.lr_nsteps)

    # train model
    model = Linear(env, Config)
    model.run(exp_schedule, lr_schedule)
