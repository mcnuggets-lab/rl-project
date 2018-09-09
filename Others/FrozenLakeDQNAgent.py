import os
from collections import deque
import numpy as np
import random
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam
import keras.backend as K


# Deep Q-learning Agent
class FrozenLakeDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.99   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.00025
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def preprocess(self, state):
        processed_state = np.zeros((16,))
        processed_state[state] = 1
        return np.array([processed_state])

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":

    episodes = 5000
    env_name = 'FrozenLake-v0'
    RENDER = False

    # initialize gym environment and the agent
    env = gym.make(env_name)
    agent = FrozenLakeDQNAgent(16, env.action_space.n)
    rewards = []

    for ep in range(episodes):

        cur_reward = 0
        state = env.reset()
        processed_state = agent.preprocess(state)
        done = False
        time = 0

        while not done:
            time += 1
            if RENDER:
                env.render()

            action = agent.act(processed_state)
            next_state, reward, done, _ = env.step(action)
            processed_next_state = agent.preprocess(next_state)
            agent.remember(processed_state, action, reward, processed_next_state, done)
            processed_state = processed_next_state
            cur_reward += reward

            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, time: {}, score: {}".format(ep, episodes, time, cur_reward))

        rewards.append(cur_reward)
        # train the agent with the experience of the episode
        if len(agent.memory) > 32:
            agent.replay(32)
        if ep > 0 and ep % 100 == 0:
            if not os.path.isdir('models/'):
                os.mkdir('models/')
            agent.model.save("./models/{}_e{}.h5".format(env_name, ep), overwrite=True)
            if not os.path.isdir('logs/'):
                os.mkdir('logs/')
            np.savetxt('./logs/{}_train.log'.format(env_name), np.array(rewards), fmt="%d")

    # do a final saving and logging after training
    if not os.path.isdir('models/'):
        os.mkdir('models/')
    agent.model.save("./models/{}.h5".format(env_name), overwrite=True)
    if not os.path.isdir('logs/'):
        os.mkdir('logs/')
    np.savetxt('./logs/{}_train.log'.format(env_name), np.array(rewards), fmt="%d")


