import curses
from arena import arena
import constants
from utils import *
from collections import deque
from time import sleep
from pause import *
from snake import Snake

"""
order of execution
1. check if the previous state produced a collision
2. if it does then do not render the next state and quit to over screen
3. if no collision, then render the state on the screen
"""


def game(screen, setting):
    if setting is None:
        return

    snake = Snake([(12, 12), (12, 11), (12, 10)])
    score = 0

    screen.nodelay(True)
    try:
        food = gen_food(snake)

        while True:

            r, c = screen.getmaxyx()
            if r < HEIGHT or c < WIDTH:
                pause(screen, r, c)
                screen.refresh()
                sleep(0.1)
                continue

            key = screen.getch()
            snake.change_direction(key)
            snake.move()

            if snake.head() != food:
                snake.removeTail()
            else:
                score += 1
                food = gen_food(snake)
            
            if snake.is_collision():
                screen.clear()
                break
            
            screen.clear()
            arena(screen, score)
            snake.draw(screen)
            draw_food(screen, food)

            screen.refresh()

            sleep(setting)
    finally:
        screen.nodelay(False)
        return score

    




        
