import gym
from gym import wrappers
from skimage.io import imshow
from skimage.color import rgb2gray


mode = 0  # 0 = deterministic, 1 = not deterministic
if mode == 0:
    env = gym.make('BreakoutDeterministic-v4')
elif mode == 1:
    env = gym.make("Breakout-v0")
else:
    raise ValueError("mode should be either 0 or 1.")

# record the episodes
env = wrappers.Monitor(env, 'records/', video_callable=lambda episode_id: True, force=True)

import numpy as np
import matplotlib.pyplot as plt
import random

num_episodes = 100 if mode == 0 else 100
frames = []
rewards = []
signed_rewards = []

for i in range(num_episodes):
    obs = env.reset()
    total_rewards = 0
    total_signed_rewards = 0
    cur_frames = []
    time = 0
    done = False
    while not done:
        time += 1

        # uncomment this line to see the agent in action
        # env.render()

        action = env.action_space.sample()
        #print(action)

        obs, reward, done, info = env.step(action)
        total_rewards += reward
        total_signed_rewards += np.sign(reward)
        cur_frames.append(obs)
    rewards.append(total_rewards)
    signed_rewards.append(total_signed_rewards)
    frames.append(cur_frames)

    print("Episode {} finished after {} timesteps with reward {}, hitted the bricks {} times.".format(
        i, time, total_rewards, total_signed_rewards))

print("{} episodes finished. Average reward is {}, and average hit is {}.".format(
    num_episodes, np.mean(rewards), np.mean(signed_rewards)))

env.close()

# rewards_bin_width = 20
# signed_rewards_bin_width = 10
# plt.hist(rewards, bins=np.arange(min(rewards), max(rewards) + rewards_bin_width, rewards_bin_width), label="score")
# plt.legend()
# plt.savefig("physics_agent_score.png")
# plt.close()
# plt.hist(signed_rewards, bins=np.arange(min(signed_rewards), max(signed_rewards) + signed_rewards_bin_width,
#                                         signed_rewards_bin_width), label="bricks hit")
# plt.legend()
# plt.savefig("physics_agent_hits.png")
# plt.close()


