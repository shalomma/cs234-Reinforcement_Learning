import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network_utils import build_mlp, device, np2torch


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """
    def __init__(self, env, config):
        """
        TODO:
        Create self.network using build_mlp, and create self.optimizer to
        optimize its parameters.
        You should find some values in the config, such as the number of layers,
        the size of the layers, and the learning rate.
        """
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate

        #######################################################
        self.baseline = build_mlp(input_size=env.observation_space.shape[0], output_size=1,
                                  n_layers=self.config.n_layers, size=self.config.layer_size)
        self.optimizer = torch.optim.Adam(self.baseline.parameters(), lr=self.lr)
        #######################################################

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]

        TODO:
        Run the network forward and then squeeze the result so that it's
        1-dimensional. Put the squeezed result in a variable called "output"
        (which will be returned).

        Note:
        A nn.Module's forward method will be invoked if you
        call it like a function, e.g. self(x) will call self.forward(x).
        When implementing other methods, you should use this instead of
        directly referencing the network (so that the shape is correct).
        """
        #######################################################
        output = self.baseline(observations).flatten()
        #######################################################
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        TODO:
        Evaluate the baseline and use the result to compute the advantages.
        Put the advantages in a variable called "advantages" (which will be
        returned).

        Note:
        The arguments and return value are numpy arrays. The np2torch function
        converts numpy arrays to torch tensors. You will have to convert the
        network output back to numpy, which can be done via the numpy() method.
        """
        observations = np2torch(observations)
        #######################################################
        advantages = (np2torch(returns) - self.baseline(observations).flatten()).detach().numpy()
        #######################################################
        return advantages

    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]

        TODO:
        Compute the loss (MSE), backpropagate, and step self.optimizer.
        You may (though not necessary) find it useful to do perform these steps
        more than one once, since this method is only called once per policy update.
        If you want to use mini-batch SGD, we have provided a helper function
        called batch_iterator (implemented in general.py).
        """
        returns = np2torch(returns)
        observations = np2torch(observations)
        #######################################################
        self.optimizer.zero_grad()
        F.mse_loss(returns.flatten(), self.baseline(observations).flatten()).backward()
        self.optimizer.step()
        #######################################################
