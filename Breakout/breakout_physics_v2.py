import gym
from gym import wrappers
from skimage.io import imshow
from skimage.color import rgb2gray

import numpy as np
import matplotlib.pyplot as plt
import random


class BreakoutPhysicsAgent:
    def __init__(self):
        self.frame = np.zeros((210, 144))
        self.last_frame = np.zeros((210, 144))
        self.ball_position = (0, 0)
        self.ball_velocity = (0, 0)
        self.paddle_position = (0, 0)
        self.blocks = np.ones((6, 18))
        self.last_predict = 36  # middle of the screen
        self.last_ball = (0, 0)
        self.last_velocity = (0, 0)
        self.last_paddle = (0, 0)

    @staticmethod
    def to_grayscale(frame):
        return (rgb2gray(frame) * 255).astype('uint8')

    def get_position(self, image):
        return np.transpose(np.where(image != 0))

    def predict_drop_pos(self, ball, velocity):
        if velocity[0] <= 0:
            # TODO: Trace the ball when it is going up.
            return self.last_predict
        else:
            drop_pos = ball[1] + velocity[1] * (39 - ball[0]) / velocity[0]
            if drop_pos < 0:
                drop_pos = -drop_pos
            elif drop_pos > 72:
                drop_pos = 72 * 2 - drop_pos
            self.last_predict = drop_pos
            return drop_pos

    def extract_features(self, frame):

        # turn the frame to grayscale
        frame = self.to_grayscale(frame)

        self.last_frame = self.frame
        self.last_ball = self.ball_position
        self.last_paddle = self.paddle_position
        self.last_velocity = self.ball_velocity

        # the useful frame area is only in the center part
        self.frame = frame[32:193, 8:152]

        # the ball is always a 4x2 rectangle
        # ball = self.frame[61:157:4, ::2]
        ball = ((self.frame[1:157:4, ::2].astype(int) - self.last_frame[1:157:4, ::2].astype(int)) > 0).astype(int)

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
            # average two recent velocity to hopefully get more stable prediction
            # if np.max(np.array(self.ball_velocity) - np.array(self.last_velocity)) <= 1:
            #     self.ball_velocity = ((self.ball_velocity[0] + self.last_velocity[0]) / 2,
            #                           (self.ball_velocity[1] + self.last_velocity[1]) / 2)

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

        return self.paddle_position, self.ball_position, self.ball_velocity

    def action(self, obs):
        paddle_pos, ball_pos, velocity = self.extract_features(obs)
        if (ball_pos[0] == ball_pos[1] == 0) or (velocity[0] == velocity[1] == 0):
            # no ball detected, so fire a new one
            act = 1
        elif velocity[0] >= 30 or velocity[1] >= 30:
            # the velocity is too fast, maybe we just started a ball?
            act = 1
        else:
            # The actions are ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
            xpos = self.predict_drop_pos(ball_pos, velocity)
            if abs(xpos) > 72:
                act = 1
            elif xpos > paddle_pos[1]:
                act = 2
            elif xpos < paddle_pos[0]:
                act = 3
            else:
                act = 1
        return act


if __name__ == "__main__":
    mode = 1  # 0 = deterministic, 1 = not deterministic
    if mode == 0:
        env = gym.make('BreakoutDeterministic-v4')
    elif mode == 1:
        env = gym.make("Breakout-v0")
    else:
        raise ValueError("mode should be either 0 or 1.")

    # record the episodes
    env = wrappers.Monitor(env, 'records/', video_callable=lambda episode_id: True, force=True)

    num_episodes = 100
    frames = []
    rewards = []
    signed_rewards = []

    for i in range(num_episodes):
        agent = BreakoutPhysicsAgent()
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

            action = agent.action(obs)
            #print(action)

            obs, reward, done, info = env.step(action)
            total_rewards += reward
            total_signed_rewards += np.sign(reward)
            # cur_frames.append(rgb2gray(obs))
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


