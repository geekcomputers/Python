from Ball import Ball
from Slab import Slab
import pygame

WIDTH = 600
HEIGHT = 600
BLACK = (0,0,0)
WHITE = (255,)*3
pygame.init()

win = pygame.display.set_mode((WIDTH, HEIGHT ))

print("Controls: W&S for player 1 and arrow up and down for player 2")

ball  = Ball([300,300 ], [0.3,0.1], win, 10, (0,0), (600,600))
slab  = Slab(win, [10,100], [500, 300], 1, (0, 0), (600, 600))
slab2 = Slab(win, [10,100], [100, 300], 2, (0, 0), (600, 600))
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    win.fill(BLACK)

    ball.borderCollisionCheck()
    ball.checkSlabCollision(slab.getCoords())
    ball.checkSlabCollision(slab2.getCoords())
    ball.updatePos()
    ball.drawBall()

    slab.updatePos()
    slab.draw()

    slab2.updatePos()
    slab2.draw()

    pygame.display.update()
pygame.quit()