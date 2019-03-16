import random as rnd
import numpy as np

class Memory:
    def __init__(self, max_memory_size, height, width, depth, frames, index_of_main_frame):
        self.max_memory_size = max_memory_size
        self.height = height
        self.width = width
        self.depth = depth
        self.mem_state= np.zeros((max_memory_size, height * width, depth))
        self.mem_state_new= np.zeros((max_memory_size, height * width, depth))
        self.mem_reward = np.zeros((max_memory_size))
        self.mem_action = np.zeros((max_memory_size))
        self.mem_end = np.zeros((max_memory_size))
        self.frames = frames
        self.index_of_main_frame = index_of_main_frame
        self.index = 0

    def save(self, state, action, reward, state_new, end):
        self.mem_state[self.index] = state
        self.mem_state_new[self.index] = state_new
        self.mem_reward[self.index] = reward
        self.mem_action[self.index] = action
        self.mem_end[self.index] = end
        self.index += 1
        if (self.index >= self.max_memory_size):
            self.index = 0

    def load(self, num):
        k = rnd.sample(range(self.index_of_main_frame, self.max_memory_size - (self.frames - self.index_of_main_frame)), num)
        action = np.empty((num, self.frames))
        reward = np.empty((num, self.frames))
        end = np.empty((num, self.frames))
        state = np.empty((num, 256, 3))
        state_new = np.empty((num, 256, 3))
        for d in range(0, num):
            i = k[d]
            state[d] = self.mem_state[i - self.index_of_main_frame]
            state_new[d] = self.mem_state_new[i - self.index_of_main_frame]
            for n in range(0, self.frames):
                index = i + (n - self.index_of_main_frame)
                action[d][n] = self.mem_action[index]
                reward[d][n] = self.mem_reward[index]
                end[d][n] = self.mem_end[index]
                if (n > 0):
                    state[d] = np.hstack([state[d], self.mem_state[index]])
                    state_new[d] = np.hstack([state_new[d], self.mem_state_new[index]])
        return state, action, reward, state_new, end