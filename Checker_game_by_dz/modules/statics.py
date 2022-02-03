"""
Author : Dhruv B Kakadiya

"""

import pygame as pg

# size of board
width, height = 800, 800
rows, cols = 8, 8
sq_size = width // cols

# colours for board
yellow = (255, 255, 0)
white = (255, 255, 255)
green = (0, 255, 0)
gray = (128, 128, 128)
red = (255, 0, 0)

# colour for for next move
black = (0, 0, 0)

crown = pg.transform.scale(pg.image.load("assets/crown.png"), (45, 25))
