import pygame
import random
import math

# Initialization
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 620
PLAYER_SPEED = 5
ENEMY_SPEED_X = 2.5
ENEMY_SPEED_Y = 40
BULLET_SPEED_Y = 10
NUMBER_OF_ENEMIES = 6
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Space Invaders")
icon = pygame.image.load("battleship.png")
pygame.display.set_icon(icon)

# Load images
background = pygame.image.load("background.png")
player_img = pygame.image.load("transport.png")
enemy_img = pygame.image.load("enemy.png")
bullet_img = pygame.image.load("bullet.png")

# Background sound
pygame.mixer.music.load("background.wav")
pygame.mixer.music.play(-1)

# Font
font = pygame.font.Font("freesansbold.ttf", 32)
game_over_font = pygame.font.Font("freesansbold.ttf", 64)

# Sounds
bullet_sound = pygame.mixer.Sound("laser.wav")
explosion_sound = pygame.mixer.Sound("explosion.wav")

# Player class
class Player:
    def __init__(self):
        self.image = player_img
        self.rect = self.image.get_rect()
        self.rect.x = 370
        self.rect.y = 480
        self.speed_x = 0

    def move(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.speed_x = -PLAYER_SPEED
        elif keys[pygame.K_RIGHT]:
            self.speed_x = PLAYER_SPEED
        else:
            self.speed_x = 0

        self.rect.x += self.speed_x
        self.rect.x = max(0, min(self.rect.x, SCREEN_WIDTH - self.rect.width))

    def draw(self):
        screen.blit(self.image, self.rect)

# Bullet class
class Bullet:
    def __init__(self):
        self.image = bullet_img
        self.rect = self.image.get_rect()
        self.speed_y = 0
        self.state = "ready"

    def fire(self, x, y):
        if self.state == "ready":
            self.state = "fire"
            bullet_sound.play()
            self.rect.centerx = x + 16
            self.rect.bottom = y + 10

    def move(self):
        if self.state == "fire":
            self.rect.y -= BULLET_SPEED_Y
            if self.rect.y <= 0:
                self.state = "ready"

    def draw(self):
        if self.state == "fire":
            screen.blit(self.image, self.rect)

# Enemy class
class Enemy:
    def __init__(self, x, y):
        self.image = enemy_img
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed_x = ENEMY_SPEED_X
        self.speed_y = ENEMY_SPEED_Y

    def move(self):
        self.rect.x += self.speed_x
        if self.rect.x <= 0 or self.rect.x >= SCREEN_WIDTH - self.rect.width:
            self.speed_x *= -1
            self.rect.y += self.speed_y

    def draw(self):
        screen.blit(self.image, self.rect)

# Collision function
def is_collision(obj1, obj2):
    distance = math.sqrt((obj1.rect.centerx - obj2.rect.centerx) ** 2 + (obj1.rect.centery - obj2.rect.centery) ** 2)
    return distance < 27

# Game loop
def game_loop():
    player = Player()
    bullet = Bullet()
    enemies = [Enemy(random.randint(0, SCREEN_WIDTH), random.randint(50, 150)) for _ in range(NUMBER_OF_ENEMIES)]
    score = 0

    running = True
    while running:
        screen.fill(BLACK)
        screen.blit(background, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bullet.fire(player.rect.centerx, player.rect.top)

        player.move()
        bullet.move()

        for enemy in enemies:
            enemy.move()
            if is_collision(enemy, bullet):
                explosion_sound.play()
                bullet.state = "ready"
                bullet.rect.y = SCREEN_HEIGHT
                score += 1
                enemy.rect.x = random.randint(0, SCREEN_WIDTH)
                enemy.rect.y = random.randint(50, 150)

        player.draw()
        bullet.draw()
        for enemy in enemies:
            enemy.draw()

        show_score(10, 10)
        pygame.display.update()

    pygame.quit()

# Show score function
def show_score(x, y):
    score_surface = font.render("Score: " + str(score), True, WHITE)
    screen.blit(score_surface, (x, y))

    pygame.display.update()

# Game over function
def game_over_text():
    over_surface = game_over_font.render("GAME OVER", True, WHITE)
    screen.blit(over_surface, (200, 250))
    pygame.display.update()

game_loop()
