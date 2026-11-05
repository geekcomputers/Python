from constants import *
import random

def gen_food(snake):
    while True:
        pos = (random.randint(3, HEIGHT-2), random.randint(2, WIDTH-2))
        if not snake.is_part_of(pos):
            return pos

def draw_food(screen, food):
    r, c = food
    screen.addstr(r, c, "*")