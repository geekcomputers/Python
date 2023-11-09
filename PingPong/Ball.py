import pygame
pygame.init()

class Ball:

    def __init__(self, pos, vel, win, rad, minCoord, maxCoord):

        self.pos = pos
        self.vel = vel
        self.win = win
        self.rad = rad
        self.minCoord = minCoord
        self.maxCoord = maxCoord


    def drawBall(self):

        pygame.draw.circle(self.win, (255,)*3, self.pos, self.rad, 0)


    def doHorizontalFlip(self):

        self.vel[0] *= -1


    def doVerticalFlip(self):

        self.vel[1] *= -1


    def borderCollisionCheck(self):

        if (self.pos[0] <= self.minCoord[0]) or (self.pos[0] >= self.maxCoord[0]):

            self.doHorizontalFlip()

        if (self.pos[1] <= self.minCoord[1]) or (self.pos[1] >= self.maxCoord[1]):

            self.doVerticalFlip()

        
    def updatePos(self):

        self.pos = [self.pos[0]+self.vel[0], self.pos[1]+self.vel[1]]


    def checkSlabCollision(self, slabPos): # slab pos = [xmin, ymin, xmax, ymax]
        if (
            self.pos[0] + self.rad > slabPos[0]
            and self.pos[0] - self.rad < slabPos[2]
            and self.pos[1] + self.rad > slabPos[1]
            and self.pos[1] - self.rad < slabPos[3]
        ):
            # Handle collision here (e.g., reverse ball's direction)
            if self.pos[0] < slabPos[0] or self.pos[0] > slabPos[2]:
                self.vel[0] *= -1
            if self.pos[1] < slabPos[1] or self.pos[1] > slabPos[3]:
                self.vel[1] *= -1