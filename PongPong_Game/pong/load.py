# ./PongPong/pong/load.py

from . import ball, paddle, rectangle
from typing import Tuple


def load_balls(win_size: Tuple, radius: float, speed: Tuple, batch=None):
    ball_x = win_size[0] / 2
    ball_y = win_size[1] / 2
    new_ball = ball.BallObject(x=ball_x, y=ball_y, radius=radius, batch=batch)
    new_ball.velocity_x, new_ball.velocity_y = speed[0], speed[1]
    return [new_ball]


def load_paddles(
    paddle_pos: Tuple, width: float, height: float, acc: Tuple, batch=None
):
    new_paddle = paddle.Paddle(
        x=paddle_pos[0], y=paddle_pos[1], width=width, height=height, batch=batch
    )
    new_paddle.rightx = new_paddle.x + width
    new_paddle.acc_left, new_paddle.acc_right = acc[0], acc[1]
    return [new_paddle]


def load_rectangles(win_size: Tuple, border: float, batch=None):
    top = rectangle.RectangleObject(
        x=0, y=win_size[1] - border, width=win_size[0], height=border, batch=batch
    )
    left = rectangle.RectangleObject(
        x=0, y=0, width=border, height=win_size[1], batch=batch
    )
    right = rectangle.RectangleObject(
        x=win_size[0] - border, y=0, width=border, height=win_size[1], batch=batch
    )
    return [left, top, right]
