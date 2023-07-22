# ./PongPong/pong/rectangle.py

import pyglet


class RectangleObject(pyglet.shapes.Rectangle):
    def __init__(self, *args, **kwargs):
        super(RectangleObject, self).__init__(*args, **kwargs)
