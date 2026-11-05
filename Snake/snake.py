from constants import *
from collections import deque
import curses

class Snake:

    def __init__(self, starting_positions):
        self.__body = deque(starting_positions)
        self.__direction = (0, 1)
    
    def is_part_of(self, point):
        return point in self.__body

    def draw(self, screen):
        for r, c in self.__body:
            try:
                screen.addstr(r, c, "@")
            except curses.error:
                pass

    def move(self):
        r, c = self.__body[0]
        new_head = (r + self.__direction[0], c + self.__direction[1])
        self.__body.appendleft(new_head)

    def removeTail(self):
        self.__body.pop()

    def change_direction(self, key):
        if   key in (W, curses.KEY_UP   ) and self.__direction != (1, 0):
            self.__direction = (-1, 0)
        elif key in (S, curses.KEY_DOWN ) and self.__direction != (-1, 0):
            self.__direction = (1, 0)
        elif key in (A, curses.KEY_LEFT ) and self.__direction != (0, 1):
            self.__direction = (0, -1)
        elif key in (D, curses.KEY_RIGHT) and self.__direction != (0, -1):
            self.__direction = (0, 1)
    
    def head(self):
        return self.__body[0]

    def is_collision(self):
        head = self.__body[0]
        r, c = head
        if r == 2 or r == HEIGHT-1 or c == 0 or c == WIDTH - 1:
            return True

        if head in list(self.__body)[1:]:
            return True

        return False