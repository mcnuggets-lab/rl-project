from __future__ import division
import argparse

import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Lambda, Input, Concatenate, Reshape, Multiply, MaxPool2D
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import Callback, FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (90, 80)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        observation = observation[30::2, ::2, :]
        img = Image.fromarray(observation)
        # img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        img = img.convert('L')
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
        return np.clip(reward, -1., 1.)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='Breakout-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(42)
env.seed(42)
nb_actions = env.action_space.n


# Next, we build our model.
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
# actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

# build constant masking layers
eye = np.eye(nb_actions)
masks = []
for i in range(nb_actions):
    mask = np.zeros((nb_actions, nb_actions))
    mask[:, i] = eye[:, i]
    masks.append(mask)
masks = np.array(masks)

InpLayer = Input(shape=input_shape)
if K.image_dim_ordering() == 'tf':
    X = Permute((2, 3, 1))(InpLayer)
elif K.image_dim_ordering() == 'th':
    X = Permute((1, 2, 3))(InpLayer)
else:
    raise RuntimeError('Unknown image_dim_ordering.')
X = Conv2D(32, 8, strides=4, activation='relu')(X)
X = Conv2D(64, 4, strides=2, activation='relu')(X)
X = Conv2D(64, 3, strides=1, activation='relu')(X)
X = MaxPool2D(2)(X)
X = Flatten()(X)
Features = Dense(3, activation='softmax')(X)
Controller = Lambda(lambda x: K.concatenate([
    K.reshape(K.zeros_like(x[:, 0]), (-1, 1)),  # to make sure action 0 is dominated
    K.reshape(-x[:, 0] - x[:, 1] - 2 * x[:, 2], (-1, 1)),
    K.reshape(x[:, 1] + x[:, 2], (-1, 1)),
    K.reshape(x[:, 0] + x[:, 2], (-1, 1))]))(Features)
Controller = Activation("softmax")(Controller)
OutLayer = Dense(512, activation='relu')(X)
OutLayer = Dense(4, activation="linear")(OutLayer)
OutLayer = Multiply()([Controller, OutLayer])
model = Model(inputs=InpLayer, outputs=OutLayer)

print(model.summary())


class DisplayFeatureLayer(Callback):
    """
    Custom callback layer to get the output of PhyLayer
    """
    def __init__(self, interval=10000):
        super(DisplayFeatureLayer, self).__init__()
        self.total_steps = 0
        self.interval = interval

    def on_step_end(self, step, logs={}):
        self.total_steps += 1
        if self.total_steps > 50000 and self.total_steps % self.interval != 0:
            # Nothing to do.
            return
        print(Features.output)


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
dqn.compile(Adam(lr=.001), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can use the built-in Keras callbacks!
    weights_filename = 'wts/dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'wts/dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    # callbacks += [DisplayFeatureLayer(interval=10000)]
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
    dqn.test(env, nb_episodes=1, visualize=False)
elif args.mode == 'test':
    weights_filename = 'wts/dqn_BreakoutDeterministic-v4_weights_10000000.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    # print(env.unwrapped.get_action_meanings())
    dqn.load_weights(weights_filename)
    dqn.training = False
    dqn.test_policy = EpsGreedyQPolicy(0.01)  # set a small epsilon for test policy to avoid getting stuck
    env = gym.wrappers.Monitor(env, "records/", video_callable=lambda episode_id: True, force=True)
    dqn.test(env, nb_episodes=10, visualize=True)
    env.close()


