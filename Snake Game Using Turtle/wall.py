"""This file creates a responsive boundary wall that adapts to the game window size."""

from turtle import Turtle, Screen
import colors

class Wall:
    """ This class creates a wall around the game screen that adjusts to its dimensions. """
    def __init__(self):
        self.screen = Screen()
        self.create_wall()

    def create_wall(self):
        """Draws a responsive game border and a header area for the scoreboard and controls."""
        width = self.screen.window_width()
        height = self.screen.window_height()

        # Calculate coordinates for the border based on screen size
        top = height / 2
        bottom = -height / 2
        left = -width / 2
        right = width / 2

        wall = Turtle()
        wall.hideturtle()
        wall.speed("fastest")
        wall.color(colors.WALL_COLOR)
        wall.penup()

        # Draw the main rectangular border
        wall.goto(left + 10, top - 10)
        wall.pendown()
        wall.pensize(10)
        wall.goto(right - 10, top - 10)
        wall.goto(right - 10, bottom + 10)
        wall.goto(left + 10, bottom + 10)
        wall.goto(left + 10, top - 10)

        # Draw a line to create a separate header section for the score and buttons
        wall.penup()
        wall.goto(left + 10, top - 70)
        wall.pendown()
        wall.pensize(5)
        wall.goto(right - 10, top - 70)

        self.screen.update()

