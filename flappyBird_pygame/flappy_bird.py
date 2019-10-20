#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:17:24 2019

@author: Mehul
"""


import math
import os
import pygame
from pygame.locals import *
from random import randint
from collections import deque



FPS = 60
ANI_SPEED = 0.18  # pixels per millisecond
W_WIDTH = 284 * 2     # BG image size: 284x512 px; tiled twice
W_HEIGHT = 512


class Bird(pygame.sprite.Sprite):

    WIDTH = 32              #   bird image width
    HEIGHT = 32             #   bird image height
    DOWN_SPEED = 0.18       #   pix per ms  -y
    UP_SPEED = 0.3          #   pix per ms  +y
    UP_DURATION = 150       #   time for which bird go up

    def __init__(self, x, y, ms_to_up, images):

        super(Bird, self).__init__()
        self.x, self.y = x, y
        self.ms_to_up = ms_to_up
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)

    def update(self, delta_frames=1):

        if self.ms_to_up > 0:
            frac_climb_done = 1 - self.ms_to_up/Bird.UP_DURATION
            self.y -= (Bird.UP_SPEED * frames_to_msec(delta_frames) *
                       (1 - math.cos(frac_climb_done * math.pi)))
            self.ms_to_up -= frames_to_msec(delta_frames)
        else:
            self.y += Bird.DOWN_SPEED * frames_to_msec(delta_frames)

    @property
    def image(self):
        # to animate bird
        if pygame.time.get_ticks() % 500 >= 250:
            return self._img_wingup
        else:
            return self._img_wingdown

    @property
    def mask(self):
        # collision detection
        if pygame.time.get_ticks() % 500 >= 250:
            return self._mask_wingup
        else:
            return self._mask_wingdown

    @property
    def rect(self):
        # return birds params
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)


class PipePair(pygame.sprite.Sprite):

    WIDTH = 80          #    width of pipe
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 3000

    def __init__(self, pipe_end_img, pipe_body_img):

        self.x = float(W_WIDTH - 1)
        self.score_counted = False

        self.image = pygame.Surface((PipePair.WIDTH, W_HEIGHT), SRCALPHA)
        self.image.convert()   # speeds up blitting
        self.image.fill((0, 0, 0, 0))
        total_pipe_body_pieces = int(
            (W_HEIGHT -                  # fill window from top to bottom
             3 * Bird.HEIGHT -             # make room for bird to fit through
             3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
            PipePair.PIECE_HEIGHT          # to get number of pipe pieces
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces)
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces

        # bottom pipe
        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, W_HEIGHT - i*PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = W_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        # top pipe
        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))

        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)

    @property
    def top_height_px(self):
        # returns top pipe's height in pix
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):

        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        # pipe is on screen or not
        return -PipePair.WIDTH < self.x < W_WIDTH

    @property
    def rect(self):
        # Get the Rect which contains this Pipe.
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, delta_frames=1):

        self.x -= ANI_SPEED * frames_to_msec(delta_frames)

    def collides_with(self, bird):

        return pygame.sprite.collide_mask(self, bird)


def load_images():


    def load_image(img_file_name):

        file_name = os.path.join('.', 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'background': load_image('background.png'),
            'pipe-end': load_image('pipe_end.png'),
            'pipe-body': load_image('pipe_body.png'),
            # images for animating the flapping bird -- animated GIFs are
            # not supported in pygame
            'bird-wingup': load_image('bird_wing_up.png'),
            'bird-wingdown': load_image('bird_wing_down.png')}


def frames_to_msec(frames, fps=FPS):

    return 1000.0 * frames / fps


def msec_to_frames(milliseconds, fps=FPS):

    return fps * milliseconds / 1000.0
'''
def gameover(display, score):
    font = pygame.font.SysFont(None,55)
    text = font.render("Game Over! Score: {}".format(score),True,(255,0,0))
    display.blit(text, [150,250])'''


def main():

    pygame.init()

    display_surface = pygame.display.set_mode((W_WIDTH, W_HEIGHT))
    pygame.display.set_caption('Flappy Bird by PMN')

    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
    images = load_images()

    # the bird stays in the same x position, so bird.x is a constant
    # center bird on screen
    bird = Bird(50, int(W_HEIGHT/2 - Bird.HEIGHT/2), 2,
                (images['bird-wingup'], images['bird-wingdown']))

    pipes = deque()

    frame_clock = 0  # this counter is only incremented if the game isn't paused
    score = 0
    done = paused = False
    while not done:
        clock.tick(FPS)

        # Handle this 'manually'.  If we used pygame.time.set_timer(),
        # pipe addition would be messed up when paused.
        if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
            pp = PipePair(images['pipe-end'], images['pipe-body'])
            pipes.append(pp)

        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                done = True
                break
            elif e.type == KEYUP and e.key in (K_PAUSE, K_p):
                paused = not paused
            elif e.type == MOUSEBUTTONUP or (e.type == KEYUP and
                    e.key in (K_UP, K_RETURN, K_SPACE)):
                bird.ms_to_up = Bird.UP_DURATION

        if paused:
            continue  # don't draw anything

        # check for collisions
        pipe_collision = any(p.collides_with(bird) for p in pipes)
        if pipe_collision or 0 >= bird.y or bird.y >= W_HEIGHT - Bird.HEIGHT:
            done = True

        for x in (0, W_WIDTH / 2):
            display_surface.blit(images['background'], (x, 0))

        while pipes and not pipes[0].visible:
            pipes.popleft()

        for p in pipes:
            p.update()
            display_surface.blit(p.image, p.rect)

        bird.update()
        display_surface.blit(bird.image, bird.rect)

        # update and display score
        for p in pipes:
            if p.x + PipePair.WIDTH < bird.x and not p.score_counted:
                score += 1
                p.score_counted = True

        score_surface = score_font.render(str(score), True, (255, 255, 255))
        score_x = W_WIDTH/2 - score_surface.get_width()/2
        display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))

        pygame.display.flip()
        frame_clock += 1
    #gameover(display_surface, score)

    print('Game over! Score: %i' % score)

    pygame.quit()


if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.
    main()