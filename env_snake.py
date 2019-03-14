import numpy as np
import random as rnd

class Snake:
    number_of_possible_actions = 3
    padding = 2
    depth = 3
    def __init__(self, height, width):
        self.height = height + 2
        self.width = width + 2
        self.board = np.zeros((3, self.height, self.width))
        self.direction = 0
        self.head = [height // 2, width // 2]

        
        for i in range(0, self.width):
            self.board[0][0][i] = 1
            self.board[0][self.height - 1][i] = 1
        
        for i in range(1, self.height):
            self.board[0][i][0] = 1
            self.board[0][i][self.width - 1] = 1

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

    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1] 

    def do_action(self, action):
        reward = -0.1
        is_ended = False
        
        if (action == 0):
            #좌회전
            self.direction -= 1
        elif (action == 2):
            #우회전
            self.direction += 1
            
        if (direction < 0):
            direction = 3
        elif (direction > 3):
            direction = 0

        

        return self.get_state(), action, reward, is_ended
