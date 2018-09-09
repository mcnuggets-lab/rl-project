import gym
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imshow
from skimage.color import rgb2gray

RENDER = False
NUM_EPISODES = 100


if __name__ == "__main__":
    env = gym.make("Pong-v0")
    env = gym.wrappers.Monitor(env, 'records/', video_callable=lambda episode_id: True, force=True)
    rewards = []
    times = []
    frames = []
    action_prob = []
    cmd = {0: 0, 1: 2, 2: 3}  # we only need these 3 commands in Pong

    for episode in range(NUM_EPISODES):
        done = False
        time = 0
        score = 0
        state = env.reset()
        # cur_frames = [state]
        while not done:
            time += 1
            if RENDER:
                env.render()

            action = np.random.randint(0, 3)
            state, reward, done, info = env.step(cmd[action])
            # cur_frames.append(state)
            score += reward

            if done:
                print('Episode: {} - Score: {}, Time: {}.'.format(episode, score, time))
                rewards.append(score)
                times.append(time)
                # frames.append(cur_frames)

    print("{} episodes tested. Average score: {}, Average time: {}.".format(NUM_EPISODES, np.mean(rewards), np.mean(times)))

    env.close()

    # plt.hist(rewards, label="score")
    # plt.legend()
    # plt.savefig("pong_pg_test_score.png")
    # plt.close()


