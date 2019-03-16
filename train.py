from environment.env_snake import Snake
from agent.agent import Agent
from agent.memory import Memory

height = 15
width = 15

if __name__ == "__main__":
    snake = Snake(height, width)

    ag = Agent(snake)

    ag.train()