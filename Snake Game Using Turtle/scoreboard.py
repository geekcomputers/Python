"""
This file manages the display of the score, high score, and game messages.
It now positions the score dynamically in the top-left corner.
"""
from turtle import Turtle, Screen
import colors

# Constants for styling and alignment
ALIGNMENT = "left"
SCORE_FONT = ("Lucida Sans", 20, "bold")
MESSAGE_FONT = ("Courier", 40, "bold")
INSTRUCTION_FONT = ("Lucida Sans", 16, "normal")

class Scoreboard(Turtle):
    """ This class maintains the scoreboard, high score, and game messages. """
    def __init__(self):
        super().__init__()
        self.screen = Screen()  # Get access to the screen object
        self.score = 0
        self.high_score = self.load_high_score()
        self.penup()
        self.hideturtle()
        self.update_scoreboard()

    def load_high_score(self):
        """Loads high score from highscore.txt. Returns 0 if not found."""
        try:
            with open("highscore.txt", mode="r") as file:
                return int(file.read())
        except (FileNotFoundError, ValueError):
            return 0

    def update_scoreboard(self):
        """Clears and rewrites the score and high score in the top-left corner."""
        self.clear()
        self.color(colors.SCORE_COLOR)
        # Dynamically calculate position to be well-placed in the header
        x_pos = -self.screen.window_width() / 2 + 30
        y_pos = self.screen.window_height() / 2 - 60
        self.goto(x_pos, y_pos)
        self.write(f"Score: {self.score} | High Score: {self.high_score}", align=ALIGNMENT, font=SCORE_FONT)

    def increase_score(self):
        """Increases score and updates the display."""
        self.score += 1
        self.update_scoreboard()

    def reset(self):
        """Checks for new high score, saves it, and resets the score."""
        if self.score > self.high_score:
            self.high_score = self.score
            with open("highscore.txt", mode="w") as file:
                file.write(str(self.high_score))
        self.score = 0
        self.update_scoreboard()

    def game_over(self):
        """Displays the Game Over message and instructions."""
        self.goto(0, 40)
        self.color(colors.GAME_OVER_COLOR)
        self.write("GAME OVER", align="center", font=MESSAGE_FONT)
        self.goto(0, -40)
        self.write("Click 'Restart' or Press 'R'", align="center", font=INSTRUCTION_FONT)

    def display_pause(self):
        """Displays the PAUSED message."""
        self.goto(0, 40)
        self.color(colors.MESSAGE_COLOR)
        self.write("PAUSED", align="center", font=MESSAGE_FONT)
        self.goto(0, -40)
        self.write("Click 'Resume' or Press 'Space'", align="center", font=INSTRUCTION_FONT)
        
    def display_start_message(self):
        """Displays the welcome message and starting instructions."""
        self.goto(0, 40)
        self.color(colors.MESSAGE_COLOR)
        self.write("SNAKE GAME", align="center", font=MESSAGE_FONT)
        self.goto(0, -40)
        self.write("Click 'Play' or an Arrow Key to Start", align="center", font=INSTRUCTION_FONT)

