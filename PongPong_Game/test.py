from pong.ball import BallObject
from pong.paddle import Paddle
from pyglet.window import key
from unittest.mock import MagicMock
import pytest

def test_ball_update_elif():
    ball = BallObject(x=50, y=95, radius=10)
    ball.velocity_x = 0
    ball.velocity_y = 10
    win_size = (100, 100)
    border = 5

    other_object = MagicMock()
    other_object.height = 10
    other_object.x = 0
    other_object.rightx = 0

    ball.update(win_size=win_size, border=border, other_object=other_object, dt=0.1)
    from track import coverage_dict
    assert coverage_dict["pong/ball.py/BallObject/update.elif"]

def test_paddle_update_elif2():
    paddle = Paddle(x=90, y=50, width=10, height=10)
    paddle.key_handler = MagicMock()
    paddle.key_handler[key.LEFT] = False
    paddle.key_handler[key.RIGHT] = False
    paddle.acc_right = 0
    paddle.acc_left = 0
    win_size = (100, 100)
    border = 5

    paddle.update(win_size=win_size, border=border, other_object=MagicMock(), dt=0.1)
    from track import coverage_dict
    assert coverage_dict["pong/paddle.py/Paddle/update.elif2"]
