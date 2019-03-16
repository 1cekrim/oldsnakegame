import numpy as np
import random as rnd

class Snake:
    number_of_possible_actions = 3
    padding = 2
    depth = 3
    length = 3
    def __init__(self, height, width):
        self.height = height + 2
        self.width = width + 2
        self.init_env()

    def init_env(self):
        self.board = np.zeros((3, self.height, self.width))
        self.direction = 0
        self.body = []
        
        for i in range(0, self.width):
            self.board[0][0][i] = 1
            self.board[0][self.height - 1][i] = 1
        
        for i in range(1, self.height):
            self.board[0][i][0] = 1
            self.board[0][i][self.width - 1] = 1

        for i in range(0, 3):
            self.body.append([self.height // 2, self.width // 2 - i])
            self.board[1][self.height // 2][self.width // 2 - i] = 1
        print(self.body)

    def get_state(self):
        h = self.height + 4
        w = self.width + 4
        reshaped = np.zeros((3, h * w))
        for i in range(0, 3):
            t = np.pad(self.board[i], ((2, 2), (2, 2)), 'constant', constant_values = (0))
            reshaped[i] = t.reshape(h * w)
        print(reshaped.T.shape)
        return reshaped.T

    def show_board(self):
        s = ''
        for i in range(0, self.height):
            for j in range(0, self.width):
                if (self.board[0][i][j] == 1):
                    s += '▦ '
                elif (self.board[1][i][j] == 1 ):
                    s += '■ '
                elif (self.board[2][i][j] == 1):
                    s += '☆ '
                else:
                    s += '　'
            s += '\n'
        
        print(s)

    def put_food(self, y, x):
        self.board[2][y][x] = 1

    def put_food_random(self):
        while True:
            y, x = rnd.randrange(1, self.height - 1), rnd.randrange(1, self.width - 1)
            if (self.board[2][y][x] == 0 and self.board[1][y][x] == 0):
                break
        self.put_food(y, x)

    

    def do_action(self, action):
        is_ended = False
        dir = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        if (action == 0):
            #좌회전
            self.direction -= 1
        elif (action == 2):
            #우회전
            self.direction += 1
            
        if (self.direction < 0):
            self.direction = 3
        elif (self.direction > 3):
            self.direction = 0
        
        head = self.body[0]
        head = [head[0] + dir[self.direction][0], head[1] + dir[self.direction][1]]
        
        i = head[0]
        j = head[1]

        print(head)
        reward = -0.1

        if (self.board[0][i][j] == 1):
            reward = -1
            is_ended = True
        elif (self.board[1][i][j] == 1):
            reward = -1
            is_ended = True
        elif (self.board[2][i][j] == 1):
            reward = 1
            self.length += 1
        else:
            reward = -0.1
        
        if (reward != 1):
            self.board[1][self.body[self.length - 1][0]][self.body[self.length - 1][1]] = 0
            del self.body[self.length - 1]
        self.body.insert(0, head)
        print(self.body)
        self.board[0][self.body[0][0]][self.body[0][1]] = 0
        self.board[1][self.body[0][0]][self.body[0][1]] = 1
        self.board[2][self.body[0][0]][self.body[0][1]] = 0
            

        return self.get_state(), action, reward, is_ended
