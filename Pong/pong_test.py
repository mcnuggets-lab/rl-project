from pong_pg import PGAgent
from pong_pg_v2 import PGAgentV2
import gym
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imshow
from skimage.color import rgb2gray

RENDER = False
NUM_EPISODES = 100
STATE_SHAPE = (80, 80)


def preprocess(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


if __name__ == "__main__":
    env = gym.make("Pong-v0")
    env = gym.wrappers.Monitor(env, 'records/', video_callable=lambda episode_id: True, force=True)
    rewards = []
    times = []
    frames = []
    action_prob = []
    cmd = {0: 0, 1: 2, 2: 3}  # we only need these 3 commands in Pong

    state_size = np.prod(STATE_SHAPE)
    agent = PGAgent(state_size, len(cmd))
    agent.load('./models/Pong-v0_pg_30300.h5')
    for episode in range(NUM_EPISODES):
        done = False
        time = 0
        score = 0
        state = env.reset()
        # cur_frames = [state]
        cur_action_prob = []
        prev_x = None
        while not done:
            time += 1
            if RENDER:
                env.render()

            cur_x = preprocess(state)
            x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
            prev_x = cur_x

            action, prob = agent.act(x)
            state, reward, done, info = env.step(cmd[action])
            # cur_frames.append(state)
            cur_action_prob.append((action, prob))
            score += reward

            if done:
                print('Episode: {} - Score: {}, Time: {}.'.format(episode, score, time))
                rewards.append(score)
                times.append(time)
                # frames.append(cur_frames)
                action_prob.append(cur_action_prob)

    print("{} episodes tested. Average score: {}, Average time: {}.".format(NUM_EPISODES, np.mean(rewards), np.mean(times)))

    env.close()

    # plt.hist(rewards, label="score")
    # plt.legend()
    # plt.savefig("pong_pg_test_score.png")
    # plt.close()


