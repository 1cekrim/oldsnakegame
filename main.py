import env_snake

height = 10
width = 10

if __name__ == "__main__":
    snake = env_snake.Snake(height, width)
    snake.put_food(1, 1)
    snake.show_board()