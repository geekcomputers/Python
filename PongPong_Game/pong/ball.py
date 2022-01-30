# ./PongPong/pong/ball.py

import pyglet
import random
from typing import Tuple


class BallObject(pyglet.shapes.Circle):
    def __init__(self, *args, **kwargs):
        super(BallObject, self).__init__(*args, **kwargs)
        self.color = (255, 180, 0)
        self.velocity_x, self.velocity_y = 0.0, 0.0

    def update(self, win_size: Tuple, border: Tuple, other_object, dt) -> None:
        speed = [
            2.37,
            2.49,
            2.54,
            2.62,
            2.71,
            2.85,
            2.96,
            3.08,
            3.17,
            3.25,
        ]  # more choices more randomness
        rn = random.choice(speed)
        newx = self.x + self.velocity_x
        newy = self.y + self.velocity_y

        if newx < border + self.radius or newx > win_size[0] - border - self.radius:
            self.velocity_x = -(self.velocity_x / abs(self.velocity_x)) * rn
        elif newy > win_size[1] - border - self.radius:
            self.velocity_y = -(self.velocity_y / abs(self.velocity_y)) * rn
        elif (newy - self.radius < other_object.height) and (
            other_object.x <= newx <= other_object.rightx
        ):
            self.velocity_y = -(self.velocity_y / abs(self.velocity_y)) * rn
        else:
            self.x = newx
            self.y = newy
