import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Flatten, Input, Lambda, Activation
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
import keras.backend as K

RENDER = False
FRESH_START = True


class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 1e-3
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        InpLayer = Input(shape=(self.state_size,))
        X = Reshape((1, 80, 80))(InpLayer)
        X = Conv2D(32, 6, strides=3, padding='same', activation='relu', kernel_initializer='he_uniform')(X)
        X= Flatten()(X)
        X = Dense(64, activation='relu')(X)
        X = Dense(32, activation='relu')(X)
        OutLayer = Dense(self.action_size, activation="softmax")(X)
        # Features = Dense(3, activation='softmax')(X)
        # OutLayer = Lambda(lambda x: K.concatenate([
        #     K.reshape(-x[:, 0] - x[:, 1] - 2 * x[:, 2], (-1, 1)),
        #     K.reshape(x[:, 1] + x[:, 2], (-1, 1)),
        #     K.reshape(x[:, 0] + x[:, 2], (-1, 1))]))(Features)
        # OutLayer = Activation('softmax')(OutLayer)
        model = Model(inputs=InpLayer, outputs=OutLayer)
        opt = Adam(lr=self.learning_rate)
        # See note regarding crossentropy in cartpole_reinforce.py
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]  # stochastic policy
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def preprocess(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


if __name__ == "__main__":
    env_name = "Pong-v0"
    env = gym.make(env_name)
    state = env.reset()
    prev_x = None
    score = 0
    episode = 1
    time = 0
    cmd = {0: 0, 1: 2, 2: 3}  # we only need these 3 commands in Pong

    state_size = 80 * 80
    action_size = env.action_space.n
    agent = PGAgent(state_size, len(cmd))
    if FRESH_START:
        with open('./logs/{}_pg.log'.format(env_name), "wb") as f:
            np.savetxt(f, np.array(["episode", "score", "time"]).reshape(1, -1), fmt='%s', delimiter=",")
    else:
        try:
           agent.load('./models/{}_pg.h5'.format(env_name))
        except:
           print("Load model failed. Will train from scratch.")
           with open('./logs/{}_pg.log'.format(env_name), "wb") as f:
               np.savetxt(f, np.array(["episode", "score", "time"]).reshape(1, -1), fmt='%s', delimiter=",")
    while True:
        if RENDER:
            env.render()

        time += 1

        cur_x = preprocess(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        prev_x = cur_x

        action, prob = agent.act(x)
        state, reward, done, info = env.step(cmd[action])
        score += reward
        agent.remember(x, action, prob, reward)

        if done:
            agent.train()
            print('Episode: {} - Score: {}, Time: {}.'.format(episode, score, time))
            with open('./logs/{}_pg.log'.format(env_name), "ab") as f:
                np.savetxt(f, np.array([episode, score, time]).reshape(1, -1), fmt="%d", delimiter=",")
            if episode > 1 and episode % 50 == 0:
                agent.save('./models/pong-v0_pg_{}.h5'.format(episode))
            episode += 1
            score = 0
            time = 0
            state = env.reset()
            prev_x = None
