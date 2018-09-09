from DQNAgent import DQNAgent

import os
import numpy as np
import random
import gym

from keras.models import load_model
import keras.backend as K

episodes = 100
env_name = 'CartPole-v0'
RENDER = False

# initialize gym environment and the agent
env = gym.make(env_name)
agent = DQNAgent(np.prod(env.observation_space.shape), env.action_space.n)
agent.model = load_model("./models/{}.h5".format(env_name))
print(agent.model.summary())
agent.epsilon = 0  # remove randomness in the learning agent
rewards = []

# Iterate the game
for ep in range(episodes):

    cur_reward = 0
    state = env.reset()
    state = np.reshape(state, [1, 4])
    done = False
    time = 0

    while not done:
        time += 1
        if RENDER:
            env.render()

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        state = next_state
        cur_reward += reward

        if done:
            # print the score and break out of the loop
            print("episode: {}/{}, time: {}, score: {}".format(ep, episodes, time, cur_reward))

    rewards.append(time)

# do a final logging after testing
print("{} episodes completed. Average score: {}".format(episodes, np.mean(rewards)))
if not os.path.isdir('logs/'):
    os.mkdir('logs/')
np.savetxt('./logs/{}_test.log'.format(env_name), np.array(rewards), fmt="%d")


