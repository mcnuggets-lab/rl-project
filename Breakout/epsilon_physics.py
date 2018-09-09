from __future__ import division

import os
import numpy as np
import gym

from rl.util import get_object_config
from rl.policy import Policy
from breakout_physics_v2 import BreakoutPhysicsAgent


# modify the LinearAnnealedPolicy to allow physics agent reading the observations
class ModifiedLinearAnnealedPolicy(Policy):
    """
    Implement the linear annealing policy

    Linear Annealing Policy computes a current threshold value and
    transfers it to an inner policy which chooses the action. The threshold
    value is following a linear function decreasing over time.
    """

    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError('Policy does not have attribute "{}".'.format(attr))

        super(ModifiedLinearAnnealedPolicy, self).__init__()

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps

    def get_current_value(self):
        """Return current annealing value
        # Returns
            Value to use in annealing
        """
        if self.agent.training:
            # Linear annealed: f(x) = ax + b.
            a = -float(self.value_max - self.value_min) / float(self.nb_steps)
            b = float(self.value_max)
            value = max(self.value_min, a * float(self.agent.step) + b)
        else:
            value = self.value_test
        return value

    def select_action(self, **kwargs):
        """Choose an action to perform
        # Returns
            Action to take (int)
        """
        if self.inner_policy.agent is None:
            self.inner_policy.agent = self.agent
        setattr(self.inner_policy, self.attr, self.get_current_value())
        return self.inner_policy.select_action(**kwargs)

    @property
    def metrics_names(self):
        """Return names of metrics
        # Returns
            List of metric names
        """
        return ['mean_{}'.format(self.attr)]

    @property
    def metrics(self):
        """Return metrics values
        # Returns
            List of metric values
        """

        return [getattr(self.inner_policy, self.attr)]

    def get_config(self):
        """Return configurations of LinearAnnealedPolicy
        # Returns
            Dict of config
        """
        config = super(ModifiedLinearAnnealedPolicy, self).get_config()
        config['attr'] = self.attr
        config['value_max'] = self.value_max
        config['value_min'] = self.value_min
        config['value_test'] = self.value_test
        config['nb_steps'] = self.nb_steps
        config['inner_policy'] = get_object_config(self.inner_policy)
        return config


class EpsilonPhysicsPolicy(Policy):
    """
    Eps Physics policy either:

    - takes the action suggested by the physics engine with probability epsilon
    - takes the action suggested by the neural network with prob (1 - epsilon)

    Note: Only works for the modified DQN Agent!
    """

    def __init__(self, eps_phy=.1, eps_ran=0):
        super(EpsilonPhysicsPolicy, self).__init__()
        self.eps_phy = eps_phy
        self.eps_ran = eps_ran
        self.physics_agent = BreakoutPhysicsAgent()
        self.agent = None

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps_phy:
            frame = self.agent.processor.frame
            action = self.physics_agent.action(frame)
        elif np.random.uniform() < self.eps_ran + self.eps_phy:
            action = np.random.random_integers(0, nb_actions - 1)
        else:
            action = np.argmax(q_values)
        return action


