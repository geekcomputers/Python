"""
This file handles the creation of food. Its placement is now controlled
by the main game logic to ensure it spawns within the correct boundaries.
"""

from turtle import Turtle
import random
import colors

class Food(Turtle):
    """ This class generates food for the snake to eat. """
    def __init__(self):
        super().__init__()
        self.shape("circle")
        self.penup()
        self.shapesize(stretch_len=0.7, stretch_wid=0.7)
        self.color(colors.FOOD_COLOR)
        self.speed("fastest")

    def refresh(self, left_wall, right_wall, bottom_wall, top_wall):
        """Moves the food to a new random position within the provided game boundaries."""
        # Add a margin so food doesn't spawn exactly on the edge
        margin = 20
        random_x = random.randint(int(left_wall) + margin, int(right_wall) - margin)
        random_y = random.randint(int(bottom_wall) + margin, int(top_wall) - margin)
        self.goto(random_x, random_y)

