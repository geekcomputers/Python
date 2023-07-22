# ./PongPong/pong/paddle.py

import pyglet
from pyglet.window import key
from typing import Tuple


class Paddle(pyglet.shapes.Rectangle):
    def __init__(self, *args, **kwargs):
        super(Paddle, self).__init__(*args, **kwargs)

        self.acc_left, self.acc_right = 0.0, 0.0
        self.rightx = 0
        self.key_handler = key.KeyStateHandler()
        self.event_handlers = [self, self.key_handler]

    def update(self, win_size: Tuple, border: float, other_object, dt):

        newlx = self.x + self.acc_left
        newrx = self.x + self.acc_right

        if self.key_handler[key.LEFT]:
            self.x = newlx
        elif self.key_handler[key.RIGHT]:
            self.x = newrx

        self.rightx = self.x + self.width

        if self.x < border:
            self.x = border
            self.rightx = self.x + self.width
        elif self.rightx > win_size[0] - border:
            self.x = win_size[0] - border - self.width
            self.rightx = self.x + self.width
