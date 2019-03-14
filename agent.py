import tensorflow as tf
import numpy as np
import random as rnd

class Q_net:
    def __init__(self, height, width, depth, number_of_possible_actions, frames):
        '''
        #self.input_data = tf.placeholder(shape = [height * width, depth], dtype = tf.int32)
        #self.input_data = tf.reshape(input_data, shape = [height, width, depth])
        a = env.get_state()
        self.input_data = a.reshape(height, width, depth)
        s = ''
        for i in range(0, height):
            for j in range(0, width):
                if (self.input_data[i][j][0] == 1):
                    s += '▦ '
                elif (self.input_data[i][j][1] == 1 ):
                    s += '■ '
                elif (self.input_data[i][j][2] == 1):
                    s += '☆ '
                else:
                    s += '　'
            s += '\n'
        print(s)
        tf.reshape가 예상대로 작동하는지 테스트하는 코드
        '''
        self.height = height
        self.width = width
        self.depth = depth
        self.number_of_possible_actions = number_of_possible_actions
        self.frames = frames

        input_depth = depth * frames
        self.input_data = tf.placeholder(shape = [-1, height * width, input_depth * frames], dtype = tf.int32)
        self.input_data = tf.reshape(self.input_data, shape = [-1, height, width, input_depth * frames])

        '''
        height, width가 들어오면
        필터1 : 5x5x32
        필터2 : 5x5x32
        필터3 : 5x5x32
        필터4 : (height - 12)x(height - 12)x256 해서 1x1x256으로 만듦
        '''

        self.cw1_size = 5
        self.cw1_depth = 32
        self.cw1 = tf.get_variable("cw1", shape = [self.cw1_size, self.cw1_size, input_depth, self.cw1_depth], initializer = tf.contrib.layers.variance_scaling_initializer())
        self.cb1 = tf.get_variable("cb1", shape = [self.cw1_depth], initializer = tf.contrib.layers.variance_scaling_initializer())
        self.conv1 = tf.nn.conv2d(self.input_data, self.cw1, strides = [1, 1, 1, 1], padding='VALID') + self.cb1
        self.conv1 = tf.nn.relu(self.conv1)

        self.cw2_size = 5
        self.cw2_depth = 64
        self.cw2 = tf.get_variable("cw2", shape = [self.cw2_size, self.cw2_size, self.cw1_depth, self.cw2_depth], initializer = tf.contrib.layers.variance_scaling_initializer())
        self.cb2 = tf.get_variable("cb2", shape = [self.cw2_depth], initializer = tf.contrib.layers.variance_scaling_initializer())
        self.conv2 = tf.nn.conv2d(self.input_data, self.cw2, strides = [1, 1, 1, 1], padding='VALID') + self.cb2
        self.conv2 = tf.nn.relu(self.conv2)

        self.cw3_size = 5
        self.cw3_depth = 128
        self.cw3 = tf.get_variable("cw3", shape = [self.cw3_size, self.cw3_size, self.cw2_depth, self.cw3_depth], initializer = tf.contrib.layers.variance_scaling_initializer())
        self.cb3 = tf.get_variable("cb3", shape = [self.cw3_depth], initializer = tf.contrib.layers.variance_scaling_initializer())
        self.conv3 = tf.nn.conv2d(self.input_data, self.cw3, strides = [1, 1, 1, 1], padding='VALID') + self.cb3
        self.conv3 = tf.nn.relu(self.conv3)
        
        self.cw4_size = self.height - 12
        self.cw4_size = 256
        self.cw4 = tf.get_variable("cw4", shape = [self.cw4_size, self.cw4_size, input_depth, self.cw4_size], initializer = tf.contrib.layers.variance_scaling_initializer())
        self.cb4 = tf.get_variable("cb4", shape = [self.cw4_size], initializer = tf.contrib.layers.variance_scaling_initializer())
        self.conv4 = tf.nn.conv2d(self.input_data, self.cw4, strides = [1, 1, 1, 1], padding='VALID') + self.cb4
        self.conv4 = tf.nn.relu(self.conv4)
        
        '''
        wa = action의 가치를 계산 하는 신경망
        wv = 게임 전체의 유리함을 판단하는 신경망
        '''

        self.wa = tf.get_variable("wa", shape = [self.cw4_size, self.number_of_possible_actions], initializer = tf.contrib.layers.variance_scaling_initializer())
        self.wv = tf.get_variable("wv", shape = [self.cw4_size, 1], initializer = tf.contrib.layers.variance_scaling_initializer())

        self.adventage = tf.matmul(self.conv4, self.wa)
        self.value = tf.matmul(self.conv4, self.wv)

        self.Q = self.value + tf.subtract(self.adventage, tf.reduce_mean(self.adventage, reduction_indices = 1, keep_dims = True))
        self.selected_action = tf.argmax(self.Q, 1)



class Memory:
    def __init__(self, max_memory_size, height, width, depth, frames, index_of_main_frame):
        self.max_memory_size = max_memory_size
        self.height = height
        self.width = width
        self.depth = depth
        self.mem_state= np.zeros((max_memory_size, height * width, depth))
        self.mem_reward = np.zeros((max_memory_size))
        self.mem_action = np.zeros((max_memory_size))
        self.frames = frames
        self.index_of_main_frame = index_of_main_frame
        self.index = 0

    def save(self, state, action, reward):
        self.mem_state[self.index] = state
        self.mem_reward[self.index] = reward
        self.mem_action[self.index] = action
        self.index += 1
        if (self.index >= self.max_memory_size):
            self.index = 0

    def load(self, num):
        k = rnd.sample(range(self.index_of_main_frame, self.max_memory_size - (self.frames - self.index_of_main_frame)), num)
        action = np.empty((num, self.frames))
        reward = np.empty((num, self.frames))
        state = np.empty((num, 256, 3))
        for d in range(0, num):
            i = k[d]
            state[d] = self.mem_state[i - self.index_of_main_frame]
            for n in range(0, self.frames):
                index = i + (n - self.index_of_main_frame)
                action[d][n] = self.mem_action[index]
                reward[d][n] = self.mem_reward[index]
                if (n > 0):
                    state[d] = np.hstack([state[d], self.mem_state[index]])
        return state, reward, action



class Agent:
    def __init__(self, env):
        self.env = env
        self.padding = env.padding
        self.depth = env.depth
        self.number_of_possible_actions = env.number_of_possible_actions
        self.Q_main = Q_net(env.height + 2 * self.padding, env.width + 2 * self.padding, self.depth, self.number_of_possible_actions, self.env)
