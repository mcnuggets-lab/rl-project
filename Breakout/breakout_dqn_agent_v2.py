from __future__ import division
import argparse

import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Lambda
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        if args.mode == 'train':
            return np.clip(reward, -1., 1.)
        else:
            return reward

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='test')
parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(42)
env.seed(42)
nb_actions = env.action_space.n


# custom initializer
def my_init(shape, seed=None, dtype=None):
    wts = VarianceScaling(scale=1.,
                           mode='fan_avg',
                           distribution='uniform',
                           seed=seed)(shape, dtype=dtype)
    wts[:, 0] = wts[:, 0] + 10
    return wts


# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Conv2D(32, 8, strides=4))
model.add(Activation('relu'))
model.add(Conv2D(64, 4, strides=2))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, strides=1))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(nb_actions-1,)))  # self implemented dueling layer
model.add(Lambda(lambda x: K.dot(x, K.variable(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).T)), output_shape=((nb_actions,))))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, enable_double_dqn=True, enable_dueling_network=False, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can use the built-in Keras callbacks!
    weights_filename = 'wts/dqn_{}_weights_v2.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'wts/dqn_' + args.env_name + '_weights_{step}_v2.h5f'
    log_filename = 'dqn_{}_log_v2.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    # if os.path.isfile(checkpoint_weights_filename):
    #     print("Loading previous weights...")
    #     dqn.load_weights(checkpoint_weights_filename)
    # elif os.path.isfile(weights_filename):
    #     print("Loading previous weights...")
    #     dqn.load_weights(weights_filename)
    dqn.fit(env, callbacks=callbacks, nb_steps=20000000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'wts/dqn_Breakout-v0_weights_13000000_v2.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    print(env.unwrapped.get_action_meanings())
    np.random.seed(None)
    env.seed(None)
    dqn.load_weights(weights_filename)
    dqn.training = False
    dqn.test_policy = EpsGreedyQPolicy(0.01)  # set a small epsilon for test policy to avoid getting stuck
    env = gym.wrappers.Monitor(env, "records/", video_callable=lambda episode_id: True, force=True)
    dqn.test(env, nb_episodes=100, visualize=False)
    env.close()
