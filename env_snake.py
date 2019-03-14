import numpy as np

class Snake:
    number_of_possible_actions = 3

    def __init__(self, height, width):
        self.height = height + 2
        self.width = width + 2
        self.board = np.zeros((self.height, self.width, 3))
        
        for i in range(0, self.width):
            self.board[0][i][0] = 1
            self.board[self.height - 1][i][0] = 1
        
        for i in range(1, self.height):
            self.board[i][0][0] = 1
            self.board[i][self.width - 1][0] = 1

    def get_state(self):
        print("reshaped board.shape : " + self.board.reshape(self.height * self.width, 3))
        return self.board.reshape(self.height * self.width, 3)

    def show_board(self):
        s = ''
        for i in range(0, self.height):
            for j in range(0, self.width):
                if (self.board[i][j][0] == 1):
                    s += '▦ '
                elif (self.board[i][j][1] == 1 ):
                    s += '■ '
                elif (self.board[i][j][2] == 1):
                    s += '☆ '
                else:
                    s += '　'
            s += '\n'
        print(s)

    def put_food(self, y, x):
        self.board[y][x][2] = 1


