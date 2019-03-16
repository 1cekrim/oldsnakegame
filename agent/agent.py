import tensorflow as tf
import numpy as np
import random as rnd
import os
from copy import deepcopy
from agent.memory import Memory
from agent.network import Q_net

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 16     #입력으로 한번에 넘겨줄 경험의 수
y = 0.99            #타겟 Q에 대한 할인 인자
start_e = 1         #무작위 행위 시작 확률
min_e = 0.1         #무작위 행위 최종 확률
de = 0.99           #e의 감소율
episodes = 1000     #에피소드의 갯수
max_episode_length = 50 #꼬리잡기 방지
path = "./model"    #모델 위치
load_model = True  #모델을 불러오기 할 것인지
model_name = "model"#모델의 이름
update_freq = 4     #얼마나 자주 훈련할 것인지
update_freq_target = 100 #target이 main과 몇번마다 같아질 것인지

jList = []
rList = []

class Agent:
    def __init__(self, env):
        self.env = env
        self.padding = env.padding
        self.depth = env.depth
        self.number_of_possible_actions = env.number_of_possible_actions
        self.height = env.height + 2 * self.padding
        self.width = env.width + 2 * self.padding

    def get_copy_var_ops(self, *, dest_scope_name="target", src_scope_name="main"):
        '''타겟네트워크에 메인네트워크의 Weight값을 복사.
        Args: dest_scope_name="target"(DQN): 'target'이라는 이름을 가진 객체를 가져옴 
        src_scope_name="main"(DQN): 'main'이라는 이름을 가진 객체를 가져옴 
        Returns: list: main의 trainable한 값들이 target의 값으로 복사된 값
        출처: https://passi0n.tistory.com/88 [웅이의 공간]''' 
        op_holder = [] 
        src_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name) 
        dest_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name) 
        for src_var, dest_var in zip(src_vars, dest_vars): 
            op_holder.append(dest_var.assign(src_var.value())) 
        return op_holder


    def save(self, sess, name):
        print("저장 시작")
        saver = tf.train.Saver()
        saver.save(sess, path + "/" + name + ".ckpt")

    def load(self, sess, name):
        print("불러오기 시작")
        if not os.path.exists(path):
            os.makedirs(path)
            print(name + ".ckpt가 없습니다")
        saver = tf.train.Saver()
        saver.restore(sess, path + "/" + name + ".ckpt")

    def play(self):
        tf.reset_default_graph()
        main_net = Q_net(self.height, self.width, self.depth, self.number_of_possible_actions)
        target_net = Q_net(self.height, self.width, self.depth, self.number_of_possible_actions)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            if load_model == True:
                self.load(sess, model_name)

            for i in range(episodes):
                self.env.init_env()
                j = 0
                s = self.env.get_state()
                while j < max_episode_length:
                    j += 1
                    act = sess.run(main_net.selected_action, feed_dict={main_net.input_data_set: [s]})[0]
                    st, a, r, end = self.env.do_action(act)
                    self.env.show_board()
                    s = st
                    if end:
                        break

    def train(self):
        tf.reset_default_graph()
        main_net = Q_net(self.height, self.width, self.depth, self.number_of_possible_actions)
        target_net = Q_net(self.height, self.width, self.depth, self.number_of_possible_actions)

        init = tf.global_variables_initializer()

        e = start_e
        steps = 0
        r_all = 0
        with tf.Session() as sess:
            sess.run(init)
            if load_model == True:
                self.load(sess, model_name)
            sess.run(self.get_copy_var_ops(dest_scope_name="target_net", src_scope_name="main_net"))

            for i in range(episodes):
                mem = Memory(1000, self.height, self.width, self.depth)
                self.env.init_env()
                j = 0
                s = self.env.get_state()
                while j < max_episode_length:
                    j += 1
                    if np.random.rand(1) < e:
                        act = np.random.randint(0, self.number_of_possible_actions)
                    else:
                        act = sess.run(main_net.selected_action, feed_dict={main_net.input_data_set: [s]})[0]
                    st, a, r, end = self.env.do_action(act)
                    mem.save(s, a, r, st, end)
                    steps += 1
                    if e > min_e:
                        e *= de
                    elif steps % update_freq == 0:
                        bs = min(batch_size, mem.max_index)
                        state_batch, action_batch, reward_batch, state_new_batch, end_batch = mem.load(bs)

                        Q1 = sess.run(main_net.selected_action, feed_dict={main_net.input_data_set: state_new_batch})
                        Q2 = sess.run(target_net.Q, feed_dict={target_net.input_data_set: state_new_batch})
                        dQ = Q2[range(bs), Q1]
                        em = []
                        for k in range(0, bs):
                            if not end_batch[k]:
                                em.append(1)
                            else:
                                em.append(0)
                        tQ = reward_batch + (y * dQ * em)
                        _ = sess.run(main_net.updateModel, \
                            feed_dict={main_net.input_data_set: state_batch, main_net.targetQ: tQ, main_net.actions: action_batch})
                    if e <= min_e and steps % update_freq_target == 0:
                        sess.run(self.get_copy_var_ops(dest_scope_name="target_net", src_scope_name="main_net"))
                    r_all += r
                    s = st
                    if end:
                        break

                jList.append(j)
                rList.append(r_all)
                if (i % (episodes // 10) == 0):
                    self.save(sess, model_name)
                if len(rList) % 10 == 0:
                    print(steps, np.mean(rList[-10:]), e)
            self.save(sess, model_name)
        print("완료: " + str(sum(rList) / episodes))





