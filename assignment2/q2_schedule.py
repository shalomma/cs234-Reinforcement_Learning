import numpy as np
from utils.test_env import EnvTest


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, n_steps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            n_steps: number of steps between the two values of eps
        """
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.n_steps = n_steps

    def update(self, t):
        """
        Updates epsilon

        Args:
            t: int
                frame number
        """
        ##############################################################
        """
        TODO: modify self.epsilon such that
            it is a linear interpolation from self.eps_begin to
            self.eps_end as t goes from 0 to self.n_steps
            For t > self.n_steps self.epsilon remains constant
        """
        ##############################################################
        t_steps = t if t < self.n_steps else self.n_steps
        self.epsilon = self.eps_begin + t_steps * (self.eps_end - self.eps_begin) / self.n_steps


class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, n_steps):
        """
        Args:
            env: gym environment
            eps_begin: float
                initial exploration rate
            eps_end: float
                final exploration rate
            n_steps: int
                number of steps taken to linearly decay eps_begin to eps_end
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, n_steps)

    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action

        Args:
            best_action: int
                best action according some policy
        Returns:
            an action
        """
        ##############################################################
        """
        TODO: with probability self.epsilon, return a random action
                else, return best_action

                you can access the environment via self.env

                you may use env.action_space.sample() to generate
                a random action
        """
        ##############################################################

        return best_action if np.random.rand(1) > self.epsilon else self.env.action_space.sample()


def test1():
    env = EnvTest((5, 5, 1))
    exp_start = LinearExploration(env, 1, 0, 10)

    found_diff = False
    for i in range(10):
        rnd_act = exp_start.get_action(0)
        if rnd_act != 0 and rnd_act is not None:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")


def test2():
    env = EnvTest((5, 5, 1))
    exp_start = LinearExploration(env, 1, 0, 10)
    exp_start.update(5)
    assert exp_start.epsilon == 0.5, "Test 2 failed"
    print("Test2: ok")


def test3():
    env = EnvTest((5, 5, 1))
    exp_start = LinearExploration(env, 1, 0.5, 10)
    exp_start.update(20)
    assert exp_start.epsilon == 0.5, "Test 3 failed"
    print("Test3: ok")


def your_test():
    """
    Use this to implement your own tests if you'd like (not required)
    """
    pass


if __name__ == "__main__":
    test1()
    test2()
    test3()
    your_test()
