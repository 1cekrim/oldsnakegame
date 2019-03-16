import tensorflow as tf
import numpy as np

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
        self.input_data_set = tf.placeholder(shape = [None, height * width, input_depth], dtype = tf.float32)
        self.input_data = tf.reshape(self.input_data_set, shape = [-1, height, width, input_depth])
        #self.input_data = self.input_data_a.reshape(-1, height * width, input_depth * frames)

        '''
        height, width가 들어오면
        필터1 : 5x5x32
        필터2 : 5x5x32
        필터3 : 5x5x32
        필터4 : (height - 12)x(height - 12)x256 해서 1x1x256으로 만듦
        '''
        self.cw1_size = 5
        self.cw1_depth = 32
        self.cw1 = tf.Variable(tf.random_normal([self.cw1_size, self.cw1_size, input_depth, self.cw1_depth], stddev=1) / np.sqrt(self.cw1_size * self.cw1_size / 2))
        self.cb1 = tf.Variable(tf.zeros([self.cw1_depth]))
        self.conv1 = tf.nn.conv2d(self.input_data, self.cw1, strides = [1, 1, 1, 1], padding='VALID') + self.cb1
        self.conv1 = tf.nn.relu(self.conv1)

        self.cw2_size = 5
        self.cw2_depth = 64
        self.cw2 = tf.Variable(tf.random_normal([self.cw2_size, self.cw2_size, self.cw1_depth, self.cw2_depth], stddev=1) / np.sqrt(self.cw2_size * self.cw2_size / 2))
        self.cb2 = tf.Variable(tf.zeros([self.cw2_depth]))
        self.conv2 = tf.nn.conv2d(self.conv1, self.cw2, strides = [1, 1, 1, 1], padding='VALID') + self.cb2
        self.conv2 = tf.nn.relu(self.conv2)

        self.cw3_size = 5
        self.cw3_depth = 128
        self.cw3 = tf.Variable(tf.random_normal([self.cw3_size, self.cw3_size, self.cw2_depth, self.cw3_depth], stddev=1) / np.sqrt(self.cw3_size * self.cw3_size / 2))
        self.cb3 = tf.Variable(tf.zeros([self.cw3_depth]))
        self.conv3 = tf.nn.conv2d(self.conv2, self.cw3, strides = [1, 1, 1, 1], padding='VALID') + self.cb3
        self.conv3 = tf.nn.relu(self.conv3)
        
        self.cw4_size = self.height - 12
        self.cw4_depth = 256
        self.cw4 = tf.Variable(tf.random_normal([self.cw4_size, self.cw4_size, self.cw3_depth, self.cw4_depth], stddev=1) / np.sqrt(self.cw4_size * self.cw4_size / 2))
        self.cb4 = tf.Variable(tf.zeros([self.cw4_depth]))
        self.conv4 = tf.nn.conv2d(self.conv3, self.cw4, strides = [1, 1, 1, 1], padding='VALID') + self.cb4
        self.conv4 = tf.nn.relu(self.conv4)
        
        '''
        wa = action의 가치를 계산 하는 신경망
        wv = 게임 전체의 유리함을 판단하는 신경망
        '''
        self.stream = tf.reshape(self.conv4, shape = [-1, self.cw4_depth])
        self.wa = tf.Variable(tf.random_normal([self.cw4_depth, self.number_of_possible_actions], stddev=1) / np.sqrt(self.cw4_depth / 2))
        self.wv = tf.Variable(tf.random_normal([self.cw4_depth, 1], stddev=1) / np.sqrt(self.cw4_depth / 2))

        self.adventage = tf.matmul(self.stream, self.wa)
        self.value = tf.matmul(self.stream, self.wv)

        self.Q = self.value + tf.subtract(self.adventage, tf.reduce_mean(self.adventage, reduction_indices = 1, keep_dims = True))
        self.selected_action = tf.argmax(self.Q, 1)

        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
        self.Q_action = tf.reduce_sum(tf.multiply(self.Q, tf.one_hot([ad for ad in range(0, self.number_of_possible_actions)], self.number_of_possible_actions)), reduction_indices=1)

        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        self.error = tf.square(self.targetQ - self.Q_action)
        self.loss = tf.reduce_mean(self.error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
