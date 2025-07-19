import os
import random

import pygame

# Initialize Pygame
pygame.init()

# Set up display
screen_width = 500
screen_height = 700
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Flappy Bird")

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Use the Images folder under the script directory
images_dir = os.path.join(script_dir, "Images")

# Debugging: Print paths
print(f"Script directory: {script_dir}")
print(f"Images directory: {images_dir}")

# Check if image files exist
required_images = ["bird.png", "pipe.png", "background.png"]
missing_files = []

for img in required_images:
    img_path = os.path.join(images_dir, img)
    if not os.path.exists(img_path):
        missing_files.append(img)
        print(f"Error: Missing {img} at {img_path}")

if missing_files:
    print(f"Please place the missing files in {images_dir}")
    pygame.quit()
    exit(1)

# Load images
bird_image = pygame.image.load(os.path.join(images_dir, "bird.png")).convert_alpha()
pipe_image = pygame.image.load(os.path.join(images_dir, "pipe.png")).convert_alpha()
background_image = pygame.image.load(os.path.join(images_dir, "background.png")).convert_alpha()

# Bird class (unchanged)
class Bird:
    def __init__(self):
        self.image = bird_image
        self.x = 50
        self.y = screen_height // 2
        self.vel = 0
        self.gravity = 1

    def update(self):
        self.vel += self.gravity
        self.y += self.vel

    def flap(self):
        self.vel = -10

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

# Pipe class (unchanged)
class Pipe:
    def __init__(self):
        self.image = pipe_image
        self.x = screen_width
        self.y = random.randint(150, screen_height - 150)
        self.vel = 5

    def update(self):
        self.x -= self.vel

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))
        screen.blit(pygame.transform.flip(self.image, False, True), (self.x, self.y - screen_height))

def main():
    clock = pygame.time.Clock()
    bird = Bird()
    pipes = [Pipe()]
    score = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird.flap()

        bird.update()
        for pipe in pipes:
            pipe.update()
            if pipe.x + pipe.image.get_width() < 0:
                pipes.remove(pipe)
                pipes.append(Pipe())
                score += 1

        screen.blit(background_image, (0, 0))
        bird.draw(screen)
        for pipe in pipes:
            pipe.draw(screen)

        pygame.display.update()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()