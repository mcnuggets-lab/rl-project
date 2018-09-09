from __future__ import division
import argparse

import os
from PIL import Image
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import RMSprop, Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


WINDOW_LENGTH = 4
INPUT_SHAPE = (6,)


class AtariProcessor(Processor):

    def __init__(self):
        self.frame = np.zeros((210, 144, 0))
        self.last_frame = np.zeros((210, 144, 0))
        self.ball_position = (0, 0)
        self.ball_velocity = (0, 0)
        self.paddle_position = (0, 0)
        self.blocks = np.ones((6, 18))
        self.last_predict = 36  # middle of the screen
        self.last_ball = (0, 0)
        self.last_paddle = (0, 0)

    def get_position(self, image):
        return np.transpose(np.where(image != 0))

    def predict_drop_pos(self, ball, velocity):
        if velocity[0] <= 0:
            # TODO: Trace the ball when it is going up.
            return self.last_predict
        else:
            drop_pos = ball[1] + velocity[1] * (24 - ball[0]) / velocity[0]
            if drop_pos < 0:
                drop_pos = -drop_pos
            elif drop_pos > 72:
                drop_pos = 72 * 2 - drop_pos
            self.last_predict = drop_pos
            return drop_pos

    def process_observation(self, frame):

        # turn the frame to grayscale
        frame = rgb2gray(frame)

        self.last_frame = self.frame
        self.last_ball = self.ball_position
        self.last_paddle = self.paddle_position

        # the useful frame area is only in the center part
        self.frame = frame[32:193, 8:152]

        # the ball is always a 4x2 rectangle
        ball = self.frame[61:157:4, ::2]

        # the paddle is always a 2x16 rectangle
        paddle = self.frame[158:160:2, ::2]

        # there are 6x18 blocks initially. Each block is a 6x8 rectangle.
        self.blocks = np.minimum(self.blocks, (self.frame[25:61:6, ::8] != 0).astype(int))

        self.ball_position = self.get_position(ball)
        if self.ball_position.size == 0:
            self.ball_position = self.last_ball
            self.ball_velocity = (0, 0)
        else:
            self.ball_position = tuple(self.ball_position[0])
            self.ball_velocity = (self.ball_position[0] - self.last_ball[0], self.ball_position[1] - self.last_ball[1])
            self.last_ball = self.ball_position

        paddle_pos = self.get_position(paddle)[:,1]
        if paddle_pos.size == 0:
            self.paddle_position = self.last_paddle
        else:
            paddle_min, paddle_max = np.min(paddle_pos), np.max(paddle_pos)
            if paddle_max - paddle_min > 8:
                # length of paddle is at most 8, so this means the ball is failing...
                if paddle_min + 1 in paddle_pos:
                    paddle_max = paddle_pos[-2]
                else:
                    paddle_min = paddle_pos[1]
            self.paddle_position = (paddle_min, paddle_max)

        return np.array([*self.paddle_position, *self.ball_position, *self.ball_velocity])

    # def process_observation(self, observation):
    #     assert observation.ndim == 3  # (height, width, channel)
    #     img = Image.fromarray(observation)
    #     img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
    #     processed_observation = np.array(img)
    #     assert processed_observation.shape == INPUT_SHAPE
    #     return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32')
        processed_batch[:, :, 0:2] /= 72
        processed_batch[:, :, 2:6:2] /= 72
        processed_batch[:, :, 3:6:2] /= 24
        return processed_batch

    def process_reward(self, reward):
        # return reward
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

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
model.add(Dense(256, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(nb_actions))
model.add(Activation('linear'))
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
               train_interval=4, enable_double_dqn=True, enable_dueling_network=True, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can use the built-in Keras callbacks!
    weights_filename = 'wts/phy_dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'wts/phy_dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'phy_dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    if os.path.isfile(checkpoint_weights_filename):
        print("Loading previous checkpoint weights...")
        dqn.load_weights(checkpoint_weights_filename)
    elif os.path.isfile(weights_filename):
        print("Loading previous weights...")
        dqn.load_weights(weights_filename)
    dqn.fit(env, callbacks=callbacks, nb_steps=20000000, log_interval=10000, nb_max_start_steps=20)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=1, nb_max_start_steps=20, visualize=False)
elif args.mode == 'test':
    weights_filename = 'wts/phy_dqn_BreakoutDeterministic-v4_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    np.random.seed(None)
    env.seed(None)
    print(env.unwrapped.get_action_meanings())
    dqn.load_weights(weights_filename)
    dqn.training = False
    dqn.test_policy = EpsGreedyQPolicy(0.01)  # set a small epsilon for test policy to avoid getting stuck
    env = gym.wrappers.Monitor(env, "records/", video_callable=lambda episode_id: True, force=True)
    dqn.test(env, nb_episodes=100, nb_max_start_steps=20, visualize=False)
    env.close()
