# ./PongPong/pongpong.py

import pyglet
from pong import load

# Variables, Considering a vertical oriented window for game
WIDTH = 600   # Game Window Width
HEIGHT = 600  # Game Window Height
BORDER = 10   # Walls Thickness/Border Thickness
RADIUS = 12   # Ball Radius
PWIDTH = 120  # Paddle Width
PHEIGHT = 15  # Paddle Height
ballspeed = (-2, -2)    # Initially ball will be falling with speed (x, y)
paddleacc = (-5, 5)   # Paddle Acceleration on both sides - left: negative acc, right: positive acc, for x-axis


class PongPongWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super(PongPongWindow, self).__init__(*args, **kwargs)

        self.win_size = (WIDTH, HEIGHT)
        self.paddle_pos = (WIDTH/2-PWIDTH/2, 0)
        self.main_batch = pyglet.graphics.Batch()
        self.walls = load.load_rectangles(self.win_size, BORDER, batch=self.main_batch)
        self.balls = load.load_balls(self.win_size, RADIUS, speed=ballspeed, batch=self.main_batch)
        self.paddles = load.load_paddles(self.paddle_pos, PWIDTH, PHEIGHT, acc=paddleacc, batch=self.main_batch)

    def on_draw(self):
        self.clear()
        self.main_batch.draw()


game_window = PongPongWindow(width=WIDTH, height=HEIGHT, caption='PongPong')
game_objects = game_window.balls + game_window.paddles

for paddle in game_window.paddles:
    for handler in paddle.event_handlers:
        game_window.push_handlers(handler)



def update(dt):
    global game_objects, game_window

    for obj1 in game_objects:
        for obj2 in game_objects:
            if obj1 is obj2:
                continue
            obj1.update(game_window.win_size, BORDER, obj2, dt)


if __name__ == '__main__':
    pyglet.clock.schedule_interval(update, 1/120.0)
    pyglet.app.run()
