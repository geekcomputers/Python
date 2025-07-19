import random
import pygame
from typing import List, Tuple

# Initialize pygame
pygame.init()

# Define colors
BLACK: Tuple[int, int, int] = (0, 0, 0)
WHITE: Tuple[int, int, int] = (255, 255, 255)
GREEN: Tuple[int, int, int] = (0, 255, 0)
RED: Tuple[int, int, int] = (255, 0, 0)
BRICK_COLOR: Tuple[int, int, int] = (56, 177, 237)

# Set up the display
SCREEN_WIDTH: int = 700
SCREEN_HEIGHT: int = 500
screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Brickout Game")

class Ball:
    """
    Represents the ball in the Brickout game.
    
    Attributes:
        screen (pygame.Surface): The game screen.
        radius (int): The radius of the ball.
        x_loc (int): The x-coordinate of the ball's center.
        y_loc (int): The y-coordinate of the ball's center.
        x_vel (int): The horizontal velocity of the ball.
        y_vel (int): The vertical velocity of the ball.
        width (int): The width of the game screen.
        height (int): The height of the game screen.
    """
    def __init__(self, screen: pygame.Surface, radius: int, x: int, y: int) -> None:
        self.screen: pygame.Surface = screen
        self.radius: int = radius
        self.x_loc: int = x
        self.y_loc: int = y
        self.x_vel: int = 7
        self.y_vel: int = 2
        self.width: int = SCREEN_WIDTH
        self.height: int = SCREEN_HEIGHT

    def get_x_vel(self) -> int:
        """Returns the horizontal velocity of the ball."""
        return self.x_vel

    def get_y_vel(self) -> int:
        """Returns the vertical velocity of the ball."""
        return self.y_vel

    def draw(self) -> None:
        """Draws the ball on the screen."""
        pygame.draw.circle(self.screen, RED, (self.x_loc, self.y_loc), self.radius)

    def update(self, paddle: 'Paddle', brick_wall: 'BrickWall') -> bool:
        """
        Updates the ball's position and handles collisions with walls, paddle, and bricks.
        
        Args:
            paddle (Paddle): The player's paddle.
            brick_wall (BrickWall): The wall of bricks.
        
        Returns:
            bool: True if the ball goes out of bounds at the bottom, False otherwise.
        """
        # Update position
        self.x_loc += self.x_vel
        self.y_loc += self.y_vel

        # Handle collisions with screen walls
        if self.x_loc <= self.radius:
            self.x_vel *= -1
        elif self.x_loc >= self.width - self.radius:
            self.x_vel *= -1
        
        if self.y_loc <= self.radius:
            self.y_vel *= -1
        elif self.y_loc >= self.height - self.radius:
            return True  # Ball went out of bounds

        # Handle collisions with bricks
        if brick_wall.collide(self):
            self.y_vel *= -1

        # Handle collision with paddle
        if (self.x_loc + self.radius >= paddle.x_loc and self.x_loc <= paddle.x_loc + paddle.width) and \
           (self.y_loc + self.radius >= paddle.y_loc and self.y_loc <= paddle.y_loc + paddle.height):
            self.y_vel *= -1

        return False

class Paddle:
    """
    Represents the player's paddle in the Brickout game.
    
    Attributes:
        screen (pygame.Surface): The game screen.
        width (int): The width of the paddle.
        height (int): The height of the paddle.
        x_loc (int): The x-coordinate of the paddle's top-left corner.
        y_loc (int): The y-coordinate of the paddle's top-left corner.
        max_width (int): The maximum x-coordinate the paddle can reach.
    """
    def __init__(self, screen: pygame.Surface, width: int, height: int, x: int, y: int) -> None:
        self.screen: pygame.Surface = screen
        self.width: int = width
        self.height: int = height
        self.x_loc: int = x
        self.y_loc: int = y
        self.max_width: int = SCREEN_WIDTH - width

    def draw(self) -> None:
        """Draws the paddle on the screen."""
        pygame.draw.rect(self.screen, BLACK, (self.x_loc, self.y_loc, self.width, self.height))

    def update(self) -> None:
        """Updates the paddle's position based on the mouse's x-coordinate."""
        x, _ = pygame.mouse.get_pos()
        if 0 <= x <= self.max_width:
            self.x_loc = x

class Brick:
    """
    Represents a single brick in the Brickout game.
    
    Attributes:
        screen (pygame.Surface): The game screen.
        width (int): The width of the brick.
        height (int): The height of the brick.
        x_loc (int): The x-coordinate of the brick's top-left corner.
        y_loc (int): The y-coordinate of the brick's top-left corner.
        is_in_group (bool): Whether the brick is part of a group.
    """
    def __init__(self, screen: pygame.Surface, width: int, height: int, x: int, y: int) -> None:
        self.screen: pygame.Surface = screen
        self.width: int = width
        self.height: int = height
        self.x_loc: int = x
        self.y_loc: int = y
        self.is_in_group: bool = False

    def draw(self) -> None:
        """Draws the brick on the screen."""
        pygame.draw.rect(self.screen, BRICK_COLOR, (self.x_loc, self.y_loc, self.width, self.height))

    def add(self, group: 'BrickWall') -> None:
        """
        Adds the brick to a group.
        
        Args:
            group (BrickWall): The group to add the brick to.
        """
        group.add(self)
        self.is_in_group = True

    def remove(self, group: 'BrickWall') -> None:
        """
        Removes the brick from a group.
        
        Args:
            group (BrickWall): The group to remove the brick from.
        """
        group.remove(self)
        self.is_in_group = False

    def is_alive(self) -> bool:
        """Returns whether the brick is part of a group."""
        return self.is_in_group

    def collide(self, ball: Ball) -> bool:
        """
        Checks if the ball collides with the brick.
        
        Args:
            ball (Ball): The ball to check collision with.
        
        Returns:
            bool: True if collision occurs, False otherwise.
        """
        return (ball.x_loc + ball.radius >= self.x_loc and ball.x_loc + ball.radius <= self.x_loc + self.width) and \
               (ball.y_loc - ball.radius >= self.y_loc and ball.y_loc - ball.radius <= self.y_loc + self.height)

class BrickWall:
    """
    Represents the wall of bricks in the Brickout game.
    
    Attributes:
        screen (pygame.Surface): The game screen.
        x (int): The x-coordinate of the top-left corner of the wall.
        y (int): The y-coordinate of the top-left corner of the wall.
        brick_width (int): The width of each brick.
        brick_height (int): The height of each brick.
        bricks (List[Brick]): The list of bricks in the wall.
    """
    def __init__(self, screen: pygame.Surface, x: int, y: int, brick_width: int, brick_height: int) -> None:
        self.screen: pygame.Surface = screen
        self.x: int = x
        self.y: int = y
        self.brick_width: int = brick_width
        self.brick_height: int = brick_height
        self.bricks: List[Brick] = []

        # Initialize bricks in a grid pattern
        current_x: int = x
        current_y: int = y
        for _ in range(3):  # 3 rows
            for _ in range(4):  # 4 columns
                self.bricks.append(Brick(screen, brick_width, brick_height, current_x, current_y))
                current_x += brick_width + (brick_width // 7)
            current_y += brick_height + (brick_height // 7)
            current_x = x

    def add(self, brick: Brick) -> None:
        """
        Adds a brick to the wall.
        
        Args:
            brick (Brick): The brick to add.
        """
        self.bricks.append(brick)

    def remove(self, brick: Brick) -> None:
        """
        Removes a brick from the wall.
        
        Args:
            brick (Brick): The brick to remove.
        """
        if brick in self.bricks:
            self.bricks.remove(brick)

    def draw(self) -> None:
        """Draws all bricks in the wall on the screen."""
        for brick in self.bricks:
            brick.draw()

    def update(self, ball: Ball) -> None:
        """
        Updates the wall by checking and removing bricks that collide with the ball.
        
        Args:
            ball (Ball): The ball to check collisions with.
        """
        for i in range(len(self.bricks) - 1, -1, -1):
            if self.bricks[i].collide(ball):
                self.bricks.pop(i)

    def has_won(self) -> bool:
        """Returns True if all bricks have been destroyed, False otherwise."""
        return len(self.bricks) == 0

    def collide(self, ball: Ball) -> bool:
        """
        Checks if the ball collides with any brick in the wall.
        
        Args:
            ball (Ball): The ball to check collisions with.
        
        Returns:
            bool: True if collision occurs, False otherwise.
        """
        for brick in self.bricks:
            if brick.collide(ball):
                return True
        return False

def main() -> None:
    """Main function to run the Brickout game."""
    # Initialize game objects
    ball: Ball = Ball(screen, 25, random.randint(1, SCREEN_WIDTH), 250)
    paddle: Paddle = Paddle(screen, 100, 20, 250, 450)
    brick_wall: BrickWall = BrickWall(screen, 25, 25, 150, 50)

    # Game state variables
    is_game_over: bool = False
    game_active: bool = True
    score: int = 0

    # Set up fonts for text rendering
    font_game_over: pygame.font.Font = pygame.font.SysFont("Comic Sans MS", 40)
    font_win: pygame.font.Font = pygame.font.SysFont("Comic Sans MS", 40)
    font_score: pygame.font.Font = pygame.font.SysFont("Comic Sans MS", 40)

    # Render text surfaces
    text_game_over: pygame.Surface = font_game_over.render("Game Over!", False, BLACK)
    text_win: pygame.Surface = font_win.render("You Win!", False, BLACK)
    text_score: pygame.Surface = font_score.render(f"Score: {score}", False, BLACK)

    # Game clock for controlling FPS
    clock: pygame.time.Clock = pygame.time.Clock()

    # Main game loop
    running: bool = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen
        screen.fill(WHITE)

        if game_active:
            # Draw and update game elements
            brick_wall.draw()

            # Update score if ball hits a brick
            if brick_wall.collide(ball):
                score += 10
            text_score = font_score.render(f"Score: {score}", False, BLACK)
            screen.blit(text_score, (300, 0))

            brick_wall.update(ball)
            paddle.draw()
            paddle.update()

            # Check game over conditions
            if ball.update(paddle, brick_wall):
                is_game_over = True
                game_active = False

            if brick_wall.has_won():
                game_active = False

            ball.draw()
        else:
            # Game over screen
            if is_game_over:
                screen.blit(text_game_over, (0, 0))
            elif brick_wall.has_won():
                screen.blit(text_win, (0, 0))
            screen.blit(text_score, (300, 0))

        # Update the display
        pygame.display.flip()

        # Control frame rate
        clock.tick(60)

    # Clean up
    pygame.quit()

if __name__ == "__main__":
    main()