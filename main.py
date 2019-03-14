from env_snake import Snake
from agent import Agent
from agent import Memory

height = 15
width = 15

if __name__ == "__main__":
    memtest = Memory(100, height + 6, width + 6, 3, 3, 1)
    snake = Snake(height, width)

    for i in range(0, 100):
        snake.put_food_random()

    snake.show_board()
    snake.do_action(0)

    snake.show_board()
    snake.do_action(1)

    snake.show_board()
    snake.do_action(2)
    snake.show_board()

    ag = Agent(snake)