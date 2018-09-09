import gym
from gym import wrappers
from skimage.io import imshow
from skimage.color import rgb2gray


env = gym.make("Pong-v0")

# record the episodes
env = wrappers.Monitor(env, 'records/', video_callable=lambda episode_id: True, force=True)

import numpy as np
import matplotlib.pyplot as plt
import random


class PongPhysicsAgent:
    def __init__(self):
        self.frame = np.zeros((40, 64))
        self.last_frame = np.zeros((40, 64))
        self.ball_position = np.zeros((2,))
        self.ball_velocity = np.zeros((2,))
        self.my_paddle_position = (0, 0)
        self.opp_paddle_position = (0, 0)
        self.last_predict = 20  # middle of the screen
        self.last_ball = np.zeros((2,))
        self.last_velocity = np.zeros((2,))
        self.my_last_paddle = (0, 0)
        self.opp_last_paddle = (0, 0)

    @staticmethod
    def to_grayscale(frame):
        return (rgb2gray(frame) * 255).astype('uint8')

    def get_position(self, image):
        return np.transpose(np.where(image != 0))

    @staticmethod
    def _drop_point(ball, velocity, x):
        drop_pos = (ball[0] + velocity[0] * (x - ball[1]) / velocity[1]) % (40 * 2)
        if drop_pos < 0:
            drop_pos = -drop_pos
        elif drop_pos > 40:
            drop_pos = 40 * 2 - drop_pos
        return drop_pos

    def predict_drop_pos(self, ball, velocity):
        if velocity[1] <= 0:
            # TODO: Trace the ball when it is going left.
            return self.last_predict
        else:
            drop_pos = self._drop_point(ball, velocity, 62)
            self.last_predict = drop_pos
            return drop_pos

    def extract_features(self, frame):

        # turn the frame to grayscale
        frame = self.to_grayscale(frame)

        # remove background
        frame[frame == 83] = 0
        frame[frame == 110] = 0

        self.last_frame = self.frame
        self.last_ball = self.ball_position
        self.last_velocity = self.ball_velocity
        self.my_last_paddle = self.my_paddle_position
        self.opp_last_paddle = self.opp_paddle_position

        # the useful frame area is only in the center part
        self.frame = frame[35:195, 16:144][::4, ::2]

        # the ball is always a 1x1 rectangle (after our downsampling)
        ball = self.get_position(self.frame[:, 2:-2])
        if ball.size == 0:
            self.ball_position = np.zeros((2,))
            self.ball_velocity = np.zeros((2,))
        else:
            self.ball_position = ball[0]
            if self.last_ball.size != 0:
                self.ball_velocity = self.ball_position - self.last_ball
            else:
                self.ball_velocity = self.last_velocity


        # the paddles are always a 2x4 rectangle (after downsampling)
        opp_paddle = np.where(self.frame[:, :2:2])[0]
        if opp_paddle.size == 0:
            self.opp_paddle_position = self.opp_last_paddle
        else:
            paddle_min, paddle_max = np.min(opp_paddle), np.max(opp_paddle)
            if paddle_max - paddle_min > 4:
                # we are winning
                if paddle_min + 1 in opp_paddle:
                    paddle_max = opp_paddle[-2]
                else:
                    paddle_min = opp_paddle[1]
            self.opp_paddle_position = (paddle_min, paddle_max)
        my_paddle = np.where(self.frame[:, -2::2])[0]
        if my_paddle.size == 0:
            self.my_paddle_position = self.my_last_paddle
        else:
            paddle_min, paddle_max = np.min(my_paddle), np.max(my_paddle)
            if paddle_max - paddle_min > 4:
                # we are losing
                if paddle_min + 1 in my_paddle:
                    paddle_max = my_paddle[-2]
                else:
                    paddle_min = my_paddle[1]
            self.my_paddle_position = (paddle_min, paddle_max)

        return self.my_paddle_position, self.opp_paddle_position, self.ball_position, self.ball_velocity

    def action(self, obs):
        my_paddle_pos, opp_paddle_pos, ball_pos, velocity = self.extract_features(obs)
        # print(my_paddle_pos, opp_paddle_pos, ball_pos, velocity)
        if (ball_pos[0] == ball_pos[1] == 0) or (velocity[0] == velocity[1] == 0):
            # no ball detected, so fire a new one
            act = 0
        elif velocity[0] >= 30 or velocity[1] >= 30:
            # the velocity is too fast, maybe we just started a ball?
            act = 0
        else:
            # The actions are ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
            xpos = agent.predict_drop_pos(ball_pos, velocity)
            if abs(xpos) > 40:
                act = 0
            elif xpos > my_paddle_pos[1]:
                act = 3
            elif xpos < my_paddle_pos[0]:
                act = 2
            else:
                act = 0
        return act


if __name__ == "__main__":
    num_episodes = 100  # average -7.94
    frames = []
    rewards = []

    for i in range(num_episodes):
        agent = PongPhysicsAgent()
        obs = env.reset()
        total_rewards = 0
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
            # cur_frames.append(rgb2gray(obs))
        rewards.append(total_rewards)
        # frames.append(cur_frames)

        print("Episode {} finished after {} timesteps with reward {}.".format(i, time, total_rewards))

    print("{} episodes finished. Average reward is {}.".format(num_episodes, np.mean(rewards)))

    env.close()

    # plt.hist(rewards, label="score")
    # plt.legend()
    # plt.savefig("physics_agent_score.png")
    # plt.close()


