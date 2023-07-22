"""
Author : Dhruv B Kakadiya

"""

import pygame as pg
from .checker_board import *
from .statics import *
from .pieces import *


class checker:
    def __init__(self, window):
        self._init()
        self.window = window

    # to update the position
    def update(self):
        self.board.draw(self.window)
        self.draw_moves(self.valid_moves)
        pg.display.update()

    def _init(self):
        self.select = None
        self.board = checker_board()
        self.turn = black
        self.valid_moves = {}

    # to reset the position
    def reset(self):
        self._init()

    # select row and column
    def selectrc(self, row, col):
        if self.select:
            result = self._move(row, col)
            if not result:
                self.select = None

        piece = self.board.get_piece(row, col)
        if (piece != 0) and (piece.color == self.turn):
            self.select = piece
            self.valid_moves = self.board.get_valid_moves(piece)
            return True
        return False

    # to move the pieces
    def _move(self, row, col):
        piece = self.board.get_piece(row, col)
        if (self.select) and (piece == 0) and (row, col) in self.valid_moves:
            self.board.move(self.select, row, col)
            skip = self.valid_moves[(row, col)]
            if skip:
                self.board.remove(skip)
            self.chg_turn()
        else:
            return False
        return True

    # to draw next possible move
    def draw_moves(self, moves):
        for move in moves:
            row, col = move
            pg.draw.circle(
                self.window,
                red,
                (col * sq_size + sq_size // 2, row * sq_size + sq_size // 2),
                15,
            )

    # for changing the turn
    def chg_turn(self):
        self.valid_moves = {}
        if self.turn == black:
            self.turn = white
        else:
            self.turn = black
