import pygame
pygame.init()

class Slab:
    def __init__(self, win, size, pos, player, minPos, maxPos):
        self.win = win
        self.size = size
        self.pos = pos
        self.player = player #player = 1 or 2
        self.minPos = minPos
        self.maxPos = maxPos

        
    def draw(self):
        pygame.draw.rect(self.win, (255, 255, 255), (self.pos[0], self.pos[1], self.size[0], self.size[1]))
        
    def getCoords(self):
        return [self.pos[0], self.pos[1], self.pos[0] + self.size[0], self.pos[1] + self.size[1]]
    
    def updatePos(self):
        keys = pygame.key.get_pressed()
        if self.player == 1:
            if keys[pygame.K_UP] and self.getCoords()[1]> self.minPos[1]:
                self.pos[1] -= 0.3
            if keys[pygame.K_DOWN] and self.getCoords()[3]< self.maxPos[1]:
                self.pos[1] += 0.3
        else:
            if keys[pygame.K_w] and self.getCoords()[1]> self.minPos[1]:
                self.pos[1] -= 0.3
            if keys[pygame.K_s] and self.getCoords()[3]< self.maxPos[1]:
                self.pos[1] += 0.3