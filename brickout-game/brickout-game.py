"""
 Pygame base template for opening a window
 
 Sample Python/Pygame Programs
 Simpson College Computer Science
 http://programarcadegames.com/
 http://simpson.edu/computer-science/
 
 Explanation video: http://youtu.be/vRB_983kUMc

-------------------------------------------------

Author for the Brickout game is Christian Bender
That includes the classes Ball, Paddle, Brick, and BrickWall.

"""

import random

# using pygame python GUI
import pygame

# Define Four Colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

pygame.init()

# Setting the width and height of the screen [width, height]
size = (700, 500)
screen = pygame.display.set_mode(size)

"""
    This is a simple Ball class for respresenting a ball 
    in the game. 
"""


class Ball(object):
    def __init__(self, screen, radius, x, y):
        self.__screen = screen
        self._radius = radius
        self._xLoc = x
        self._yLoc = y
        self.__xVel = 7
        self.__yVel = 2
        w, h = pygame.display.get_surface().get_size()
        self.__width = w
        self.__height = h

    def getXVel(self):
        return self.__xVel

    def getYVel(self):
        return self.__yVel

    def draw(self):
        """
        draws the ball onto screen.
        """
        pygame.draw.circle(screen, (255, 0, 0), (self._xLoc, self._yLoc), self._radius)

    def update(self, paddle, brickwall):
        """
        moves the ball at the screen.
        contains some collision detection.
        """
        self._xLoc += self.__xVel
        self._yLoc += self.__yVel
        # left screen wall bounce
        if self._xLoc <= self._radius:
            self.__xVel *= -1
        # right screen wall bounce
        elif self._xLoc >= self.__width - self._radius:
            self.__xVel *= -1
        # top wall bounce
        if self._yLoc <= self._radius:
            self.__yVel *= -1
        # bottom drop out
        elif self._yLoc >= self.__width - self._radius:
            return True

        # for bouncing off the bricks.
        if brickwall.collide(self):
            self.__yVel *= -1

        # collision detection between ball and paddle
        paddleY = paddle._yLoc
        paddleW = paddle._width
        paddleH = paddle._height
        paddleX = paddle._xLoc
        ballX = self._xLoc
        ballY = self._yLoc

        if ((ballX + self._radius) >= paddleX and ballX <= (paddleX + paddleW)) and (
            (ballY + self._radius) >= paddleY and ballY <= (paddleY + paddleH)
        ):
            self.__yVel *= -1

        return False


"""
    Simple class for representing a paddle
"""


class Paddle(object):
    def __init__(self, screen, width, height, x, y):
        self.__screen = screen
        self._width = width
        self._height = height
        self._xLoc = x
        self._yLoc = y
        w, h = pygame.display.get_surface().get_size()
        self.__W = w
        self.__H = h

    def draw(self):
        """
        draws the paddle onto screen.
        """
        pygame.draw.rect(
            screen, (0, 0, 0), (self._xLoc, self._yLoc, self._width, self._height), 0
        )

    def update(self):
        """
        moves the paddle at the screen via mouse
        """
        x, y = pygame.mouse.get_pos()
        if x >= 0 and x <= (self.__W - self._width):
            self._xLoc = x


"""
    This class represents a simple Brick class.
    For representing bricks onto screen.
"""


class Brick(pygame.sprite.Sprite):
    def __init__(self, screen, width, height, x, y):
        self.__screen = screen
        self._width = width
        self._height = height
        self._xLoc = x
        self._yLoc = y
        w, h = pygame.display.get_surface().get_size()
        self.__W = w
        self.__H = h
        self.__isInGroup = False

    def draw(self):
        """
        draws the brick onto screen.
        color: rgb(56, 177, 237)
        """
        pygame.draw.rect(
            screen,
            (56, 177, 237),
            (self._xLoc, self._yLoc, self._width, self._height),
            0,
        )

    def add(self, group):
        """
        adds this brick to a given group.
        """
        group.add(self)
        self.__isInGroup = True

    def remove(self, group):
        """
        removes this brick from the given group.
        """
        group.remove(self)
        self.__isInGroup = False

    def alive(self):
        """
        returns true when this brick belongs to the brick wall.
        otherwise false
        """
        return self.__isInGroup

    def collide(self, ball):
        """
        collision detection between ball and this brick
        """
        brickX = self._xLoc
        brickY = self._yLoc
        brickW = self._width
        brickH = self._height
        ballX = ball._xLoc
        ballY = ball._yLoc
        ballXVel = ball.getXVel()
        ballYVel = ball.getYVel()

        if (
            (ballX + ball._radius) >= brickX
            and (ballX + ball._radius) <= (brickX + brickW)
        ) and (
            (ballY - ball._radius) >= brickY
            and (ballY - ball._radius) <= (brickY + brickH)
        ):
            return True
        else:
            return False


"""
    This is a simple class for representing a 
    brick wall.
"""


class BrickWall(pygame.sprite.Group):
    def __init__(self, screen, x, y, width, height):
        self.__screen = screen
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._bricks = []

        X = x
        Y = y
        for i in range(3):
            for j in range(4):
                self._bricks.append(Brick(screen, width, height, X, Y))
                X += width + (width / 7.0)
            Y += height + (height / 7.0)
            X = x

    def add(self, brick):
        """
        adds a brick to this BrickWall (group)
        """
        self._bricks.append(brick)

    def remove(self, brick):
        """
        removes a brick from this BrickWall (group)
        """
        self._bricks.remove(brick)

    def draw(self):
        """
        draws all bricks onto screen.
        """
        for brick in self._bricks:
            if brick != None:
                brick.draw()

    def update(self, ball):
        """
        checks collision between ball and bricks.
        """
        for i in range(len(self._bricks)):
            if (self._bricks[i] != None) and self._bricks[i].collide(ball):
                self._bricks[i] = None

        # removes the None-elements from the brick list.
        for brick in self._bricks:
            if brick is None:
                self._bricks.remove(brick)

    def hasWin(self):
        """
        Has player win the game?
        """
        return len(self._bricks) == 0

    def collide(self, ball):
        """
        check collisions between the ball and
        any of the bricks.
        """
        for brick in self._bricks:
            if brick.collide(ball):
                return True
        return False


# The game objects ball, paddle and brick wall
ball = Ball(screen, 25, random.randint(1, 700), 250)
paddle = Paddle(screen, 100, 20, 250, 450)
brickWall = BrickWall(screen, 25, 25, 150, 50)

isGameOver = False  # determines whether game is lose
gameStatus = True  # game is still running

score = 0  # score for the game.

pygame.display.set_caption("Brickout-game")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# for displaying text in the game
pygame.font.init()  # you have to call this at the start,
# if you want to use this module.

# message for game over
mgGameOver = pygame.font.SysFont("Comic Sans MS", 40)

# message for winning the game.
mgWin = pygame.font.SysFont("Comic Sans MS", 40)

# message for score
mgScore = pygame.font.SysFont("Comic Sans MS", 40)

textsurfaceGameOver = mgGameOver.render("Game Over!", False, (0, 0, 0))
textsurfaceWin = mgWin.render("You win!", False, (0, 0, 0))
textsurfaceScore = mgScore.render("score: " + str(score), False, (0, 0, 0))

# -------- Main Program Loop -----------
while not done:
    # --- Main event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # --- Game logic should go here

    # --- Screen-clearing code goes here

    # Here, we clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.

    # If you want a background image, replace this clear with blit'ing the
    # background image.
    screen.fill(WHITE)

    # --- Drawing code should go here

    """
        Because I use OOP in the game logic and the drawing code,
        are both in the same section.
    """
    if gameStatus:

        # first draws ball for appropriate displaying the score.
        brickWall.draw()

        # for counting and displaying the score
        if brickWall.collide(ball):
            score += 10
        textsurfaceScore = mgScore.render("score: " + str(score), False, (0, 0, 0))
        screen.blit(textsurfaceScore, (300, 0))

        # after scoring. because hit bricks are removed in the update-method
        brickWall.update(ball)

        paddle.draw()
        paddle.update()

        if ball.update(paddle, brickWall):
            isGameOver = True
            gameStatus = False

        if brickWall.hasWin():
            gameStatus = False

        ball.draw()

    else:  # game isn't running.
        if isGameOver:  # player lose
            screen.blit(textsurfaceGameOver, (0, 0))
            textsurfaceScore = mgScore.render("score: " + str(score), False, (0, 0, 0))
            screen.blit(textsurfaceScore, (300, 0))
        elif brickWall.hasWin():  # player win
            screen.blit(textsurfaceWin, (0, 0))
            textsurfaceScore = mgScore.render("score: " + str(score), False, (0, 0, 0))
            screen.blit(textsurfaceScore, (300, 0))

    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # --- Limit to 60 frames per second
    clock.tick(60)

# Close the window and quit.
pygame.quit()
