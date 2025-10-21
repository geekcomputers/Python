"""
This file is responsible for creating the snake and managing its movement,
extension, and reset functionality.
"""
from turtle import Turtle
import colors

STARTING_POSITIONS = [(0, 0), (-20, 0), (-40, 0)]
MOVE_DISTANCE = 20
UP, DOWN, LEFT, RIGHT = 90, 270, 180, 0

class Snake:
    """ This class creates a snake body and contains methods for movement and extension. """
    def __init__(self):
        self.segments = []
        self.create_snake()
        self.head = self.segments[0]

    def create_snake(self):
        """ Creates the initial snake body. """
        for position in STARTING_POSITIONS:
            self.add_segment(position)
        self.segments[0].color(colors.FIRST_SEGMENT_COLOR)

    def add_segment(self, position):
        """ Adds a new segment to the snake. """
        new_segment = Turtle(shape="square")
        new_segment.penup()
        new_segment.goto(position)
        new_segment.color(colors.BODY_COLOR)
        self.segments.append(new_segment)

    def extend(self):
        """ Adds a new segment to the snake's tail. """
        self.add_segment(self.segments[-1].position())
        self.segments[0].color(colors.FIRST_SEGMENT_COLOR)

    def move(self):
        """ Moves the snake forward by moving each segment to the position of the one in front."""
        for i in range(len(self.segments) - 1, 0, -1):
            x = self.segments[i - 1].xcor()
            y = self.segments[i - 1].ycor()
            self.segments[i].goto(x, y)
        self.head.forward(MOVE_DISTANCE)

    def reset(self):
        """Hides the old snake and creates a new one for restarting the game."""
        for segment in self.segments:
            segment.hideturtle()
        self.segments.clear()
        self.create_snake()
        self.head = self.segments[0]

    def up(self):
        """Turns the snake's head upwards, preventing it from reversing."""
        if self.head.heading() != DOWN:
            self.head.setheading(UP)

    def down(self):
        """Turns the snake's head downwards, preventing it from reversing."""
        if self.head.heading() != UP:
            self.head.setheading(DOWN)

    def left(self):
        """Turns the snake's head to the left, preventing it from reversing."""
        if self.head.heading() != RIGHT:
            self.head.setheading(LEFT)

    def right(self):
        """Turns the snake's head to the right, preventing it from reversing."""
        if self.head.heading() != LEFT:
            self.head.setheading(RIGHT)

