import random as rnd
import numpy as np

class Memory:
    def __init__(self, max_memory_size, height, width, depth):
        self.max_memory_size = max_memory_size
        self.height = height
        self.width = width
        self.depth = depth
        self.mem_state= np.zeros((max_memory_size, height * width, depth))
        self.mem_state_new= np.zeros((max_memory_size, height * width, depth))
        self.mem_reward = np.zeros(max_memory_size)
        self.mem_action = np.zeros(max_memory_size)
        self.mem_end = np.zeros(max_memory_size)
        self.index = 0
        self.max_index = 0

    def save(self, state, action, reward, state_new, end):
        self.mem_state[self.index] = state
        self.mem_state_new[self.index] = state_new
        self.mem_reward[self.index] = reward
        self.mem_action[self.index] = action
        self.mem_end[self.index] = end
        self.index += 1
        if self.max_index < self.max_memory_size:
            self.max_index += 1 
        if (self.index >= self.max_memory_size):
            self.index = 0

    def load(self, num):
        if num > self.max_index:
            num = self.max_index
        k = rnd.sample(range(0, self.max_index), num)
        action = np.empty(num)
        reward = np.empty(num)
        end = np.empty(num)
        state = np.empty((num, self.height * self.width, self.depth))
        state_new = np.empty((num, self.height * self.width, self.depth))
        for d in range(0, num):
            state[d] = self.mem_state[k[d]]
            state_new[d] = self.mem_state_new[k[d]]
            action[d] = self.mem_action[k[d]]
            reward[d] = self.mem_reward[k[d]]
            end[d] = self.mem_end[k[d]]
        return state, action, reward, state_new, end