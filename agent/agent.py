import tensorflow as tf
import numpy as np
import random as rnd

from agent.memory import Memory
from agent.network import Q_net

batch_size = 16     #입력으로 한번에 넘겨줄 경험의 수
y = 0.99            #타겟 Q에 대한 할인 인자
e = 1               #무작위 행위 확률
min_e = 0.1         #무작위 행위 최종 확률
de = 0.99           #e의 감소율
episodes = 1000     #에피소드의 갯수
max_episode_length = 50 #꼬리잡기 방지
path = "./model"    #모델 위치
frames = 3          #한번에 넘겨줄 프레임의 수
main_frame = 1      #현재 상태의 프레임의 index


class Agent:
    def __init__(self, env):
        self.env = env
        self.padding = env.padding
        self.depth = env.depth
        self.number_of_possible_actions = env.number_of_possible_actions
        self.height = env.height + 2 * self.padding
        self.width = env.width + 2 * self.padding
    
    def train(self):
        tf.reset_default_graph()
        main_net = Q_net(self.height, self.width, self.depth, self.number_of_possible_actions, frames)
        target_net = Q_net(self.height, self.width, self.depth, self.number_of_possible_actions, frames)

        init = tf.global_variables()

        saver = tf.train.Saver()

