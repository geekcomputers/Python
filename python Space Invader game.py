import pygame
import random
import math
from pygame import mixer

# initialization

pygame.init()

# create the screen
screen = pygame.display.set_mode((800, 620))

# background

background = pygame.image.load('background.png')

#bg sound
mixer.music.load('background.wav')
mixer.music.play(-1)

# title and icon
pygame.display.set_caption("Space Invendera")
icon = pygame.image.load('battleship.png')
pygame.display.set_icon(icon)

# player
playerimg = pygame.image.load('transport.png')
playerx = 370
playery = 480
playerx_change = 0

# enemy
enemyimg = []
enemyx = []
enemyy = []
enemyx_change = []
enemyy_change = []
number_of_enemies = 6

for i in range(number_of_enemies):
    enemyimg.append(pygame.image.load('enemy.png'))
    enemyx.append(random.randint(0, 800))
    enemyy.append(random.randint(50, 150))
    enemyx_change.append(2.5)
    enemyy_change.append(40)

# bullet
bulletimg = pygame.image.load('bullet.png')
bulletx = 0
bullety = 480
bulletx_change = 0
bullety_change = 10
bullet_state = "ready"

#score
score_value = 0
font = pygame.font.Font('freesansbold.ttf',32)
textx = 10
texty = 10

#game over txt
over_font = pygame.font.Font('freesansbold.ttf',64)

def show_score(x ,y):
    score = font.render("score :"+ str(score_value),True, (255, 255, 255))
    screen.blit(score, (x, y))

def game_over_text():
    over_txt = over_font.render("GAME OVER", True, (255, 255, 255))
    screen.blit(over_txt, (200, 250))

# for display player img
def player(x, y):
    screen.blit(playerimg, (x, y))


# foe desplaing enemy img

def enemy(x, y ,i):
    screen.blit(enemyimg[i], (x, y))


def fire_bullet(x, y):
    global bullet_state
    bullet_state = "fire"
    screen.blit(bulletimg, (x + 16, y + 10))


def iscollision(enemyx, enemyy, bulletx, bullety):
    distance = math.sqrt((math.pow(enemyx - bulletx, 2)) + (math.pow(enemyy - bullety, 2)))
    if distance < 27:
        return True
    else:
        return False


# game loop
running = True
while running:

    screen.fill((0, 0, 0))
    # for bg img
    screen.blit(background, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # if keystroke in pressed whether it is right of left
        if (event.type == pygame.KEYDOWN):
            if (event.key == pygame.K_LEFT):
                playerx_change = -5
            if (event.key == pygame.K_RIGHT):
                playerx_change = 5

            if (event.key == pygame.K_SPACE):
                if bullet_state == "ready":
                    bullet_sound = mixer.Sound('laser.wav')
                    bullet_sound.play()
                    bulletx = playerx
                    fire_bullet(bulletx, bullety)

        if (event.type == pygame.KEYUP):
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                playerx_change = 0

    playerx += playerx_change
    # create boundry for player
    if playerx <= 0:
        playerx = 0
    elif playerx >= 736:
        playerx = 736

    for i in range(number_of_enemies):

        #game over
        if enemyy[i] > 440:
            for j in range(number_of_enemies):
                enemyy[j] = 2000
            game_over_text()
            break

        enemyx[i] += enemyx_change[i]
        # create boundry for enemy
        if enemyx[i] <= 0:
            enemyx_change[i] = 2.5
            enemyy[i] += enemyy_change[i]
        elif enemyx[i] >= 736:
            enemyx_change[i] = -2.5
            enemyy[i] += enemyy_change[i]

        # collision
        collision = iscollision(enemyx[i], enemyy[i], bulletx, bullety)
        if collision:
            explossion_sound = mixer.Sound('explosion.wav')
            explossion_sound.play()
            bullety = 480
            bullet_state = "ready"
            score_value += 1
            enemyx[i] = random.randint(0, 800)
            enemyy[i] = random.randint(50, 150)

        enemy(enemyx[i], enemyy[i], i)

    # bullet movement
    if bullety <= 0:
        bullety = 480
        bullet_state = "ready"

    if bullet_state == "fire":
        fire_bullet(bulletx, bullety)
        bullety -= bullety_change

    player(playerx, playery)
    show_score(textx,texty)
    pygame.display.update()
