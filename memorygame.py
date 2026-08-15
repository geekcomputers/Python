#!/usr/bin/env python3
"""
Memory Game – a tile‑matching game using the Turtle graphics library.

The board consists of 64 tiles (8x8 grid), each hiding one of 32 possible
icons (numbers 0–31). The player clicks on tiles to reveal them. When two
consecutive revealed tiles show the same icon, they remain visible; otherwise
they are hidden again. The game continues until all pairs are found.

Dependencies:
    - Python standard library: random, turtle
    - freegames (optional) – provides a car image; if not available, remove
      the car-related lines and the game will work without it.

Author: Adapted from a freegames example.
Date: 2026-07-12
"""

import random
import turtle

from freegames import path

# ----------------------------------------------------------------------------
# Game constants and global state
# ----------------------------------------------------------------------------

# Load the car image (used as a decorative stamp on the board)
car = path("car.gif")

# Create a list of 64 tile values: each number from 0 to 31 appears exactly twice
tiles = list(range(32)) * 2

# State dictionary: 'mark' stores the index of the currently selected tile
# (the first tile clicked in a pair attempt). None means no tile is selected.
state = {"mark": None}

# Boolean list indicating whether each tile is hidden (True = hidden, False = shown)
hide = [True] * 64


# ----------------------------------------------------------------------------
# Drawing utilities
# ----------------------------------------------------------------------------


def square(x: float, y: float) -> None:
    """
    Draw a 50x50 white square with a black outline at the given coordinates.

    The turtle starts at the bottom‑left corner of the square and draws it
    counter‑clockwise using the standard forward/left movements.

    Args:
        x: X‑coordinate of the bottom‑left corner.
        y: Y‑coordinate of the bottom‑left corner.
    """
    turtle.up()
    turtle.goto(x, y)
    turtle.down()
    turtle.color("black", "white")
    turtle.begin_fill()
    for _ in range(4):
        turtle.forward(50)
        turtle.left(90)
    turtle.end_fill()


# ----------------------------------------------------------------------------
# Coordinate and index conversion
# ----------------------------------------------------------------------------


def index(x: float, y: float) -> int:
    """
    Convert screen coordinates to a tile index (0‑63).

    The board is arranged in an 8x8 grid, with each cell being 50x50 pixels.
    The origin (0,0) is at the centre of the screen; the board occupies the
    region from (-200, -200) to (200, 200). This function maps a click
    position to the corresponding tile index, row‑major order.

    Args:
        x: X‑coordinate of the click (in turtle screen units).
        y: Y‑coordinate of the click.

    Returns:
        int: Tile index (0 to 63).
    """
    return int((x + 200) // 50 + ((y + 200) // 50) * 8)


def xy(count: int) -> tuple[float, float]:
    """
    Convert a tile index to screen coordinates of its bottom‑left corner.

    This is the inverse of `index()`. The calculation uses integer division
    and modulo to determine the row and column.

    Args:
        count: Tile index (0‑63).

    Returns:
        tuple[float, float]: (x, y) coordinates of the tile's bottom‑left corner.
    """
    return (count % 8) * 50 - 200, (count // 8) * 50 - 200


# ----------------------------------------------------------------------------
# Game logic
# ----------------------------------------------------------------------------


def tap(x: float, y: float) -> None:
    """
    Handle a mouse click event on the canvas.

    This is the core game logic. It performs the following steps:
        1. Determine which tile was clicked.
        2. If no tile is currently marked (state["mark"] is None),
           mark this tile as the first of a pair.
        3. If a tile is already marked and it is not the same tile,
           compare the values of the marked tile and the newly clicked tile:
             - If they match (same number), reveal both tiles permanently.
             - If they do not match, unmark the tile (keep it hidden).
        4. If the same tile is clicked twice, it simply remains marked.

    Args:
        x: X‑coordinate of the click.
        y: Y‑coordinate of the click.
    """
    spot = index(x, y)
    mark = state["mark"]

    # If no mark, or we clicked the same tile again, or the values differ:
    # just set the mark to the new spot (or keep it if it's the same).
    # Note: if the two values are different, we don't hide anything – the
    # tile will be drawn hidden again in the next draw() cycle because
    # hide[spot] remains True.
    if mark is None or mark == spot or tiles[mark] != tiles[spot]:
        state["mark"] = spot
    else:
        # Values match: reveal both tiles permanently.
        hide[spot] = False
        hide[mark] = False
        # Clear the mark so the next click starts a new pair.
        state["mark"] = None


def draw() -> None:
    """
    Redraw the entire board and schedule the next animation frame.

    This function is called repeatedly via turtle.ontimer() to update the
    display. It:
        - Clears the canvas.
        - Draws the decorative car image at the centre.
        - Draws a white square for every hidden tile.
        - If a tile is marked and currently hidden, it writes the tile's
          value (number) on top of that square.
        - Finally, triggers a redraw after 100 milliseconds.

    The use of turtle.tracer(False) and manual turtle.update() ensures
    smooth animation without flickering.
    """
    turtle.clear()
    turtle.goto(0, 0)
    turtle.shape(car)
    turtle.stamp()  # Place the car image permanently on the canvas

    # Draw all tiles that are still hidden.
    for count in range(64):
        if hide[count]:
            x, y = xy(count)
            square(x, y)

    # If a tile is marked and hidden, display its value.
    mark = state["mark"]
    if mark is not None and hide[mark]:
        x, y = xy(mark)
        turtle.up()
        turtle.goto(x + 2, y)  # Shift slightly for better centering
        turtle.color("black")
        turtle.write(tiles[mark], font=("Arial", 30, "normal"))

    turtle.update()  # Refresh the screen with all drawn elements
    turtle.ontimer(draw, 100)  # Schedule next redraw (10 fps)


# ----------------------------------------------------------------------------
# Main setup and execution
# ----------------------------------------------------------------------------

# Shuffle the tile values to randomise the board layout.
random.shuffle(tiles)

# Configure the turtle window.
turtle.setup(420, 420, 370, 0)  # Window size, start x, start y
turtle.addshape(car)  # Register the car image as a shape
turtle.hideturtle()  # Hide the turtle cursor (we only need drawings)
turtle.tracer(False)  # Disable automatic updates for performance
turtle.onscreenclick(tap)  # Bind mouse clicks to the tap() function

# Start the game loop.
draw()

# Keep the window open (this is the main event loop).
turtle.done()
