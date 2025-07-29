import random
import tkinter as tk

# Game Constants
WIDTH: int = 600
HEIGHT: int = 400
BALL_RADIUS: int = 20
PAD_WIDTH: int = 8
PAD_HEIGHT: int = 80
HALF_PAD_WIDTH: int = PAD_WIDTH // 2
HALF_PAD_HEIGHT: int = PAD_HEIGHT // 2
LEFT: bool = False
RIGHT: bool = True

Vector2D = tuple[float, float]  # Type alias for 2D vector


class PongGame:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pong Game")
        self.root.resizable(False, False)

        # Game state
        self.score1: int = 0
        self.score2: int = 0
        self.paddle1_pos: float = 0.0
        self.paddle2_pos: float = 0.0
        self.paddle1_vel: float = 0.0
        self.paddle2_vel: float = 0.0
        self.ball_pos: list[float] = [WIDTH / 2, HEIGHT / 2]
        self.ball_vel: list[float] = [0.0, 0.0]

        # Game UI components
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas.pack()

        self.restart_button = tk.Button(root, text="Restart", command=self.reset_game)
        self.restart_button.pack()

        # Initialize game elements
        self.ball = self.canvas.create_oval(
            self.ball_pos[0] - BALL_RADIUS,
            self.ball_pos[1] - BALL_RADIUS,
            self.ball_pos[0] + BALL_RADIUS,
            self.ball_pos[1] + BALL_RADIUS,
            fill="white",
        )

        self.paddle1 = self.canvas.create_line(
            PAD_WIDTH / 2, 0, PAD_WIDTH / 2, PAD_HEIGHT, width=PAD_WIDTH, fill="white"
        )

        self.paddle2 = self.canvas.create_line(
            WIDTH - PAD_WIDTH / 2,
            0,
            WIDTH - PAD_WIDTH / 2,
            PAD_HEIGHT,
            width=PAD_WIDTH,
            fill="white",
        )

        self.score_text1 = self.canvas.create_text(
            250, 30, text="0", fill="white", font=("Arial", 40)
        )
        self.score_text2 = self.canvas.create_text(
            350, 30, text="0", fill="white", font=("Arial", 40)
        )

        # Setup event handlers
        self.root.bind("<KeyPress>", self.handle_key_press)
        self.root.bind("<KeyRelease>", self.handle_key_release)

        # Start game loop
        self.game_loop()

    def spawn_ball(self, direction: bool) -> None:
        """Spawn the ball with initial velocity based on direction (LEFT/RIGHT)."""
        self.ball_pos = [WIDTH / 2, HEIGHT / 2]
        speed_x = random.randrange(120, 240) / 60
        speed_y = random.randrange(60, 180) / 60

        self.ball_vel = [speed_x if direction == RIGHT else -speed_x, speed_y]

    def reset_game(self) -> None:
        """Reset game state to initial conditions."""
        self.score1 = 0
        self.score2 = 0
        self.paddle1_pos = 0
        self.paddle2_pos = 0
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.update_score_display()
        self.spawn_ball(RIGHT)

    def update_score_display(self) -> None:
        """Update score display on canvas."""
        self.canvas.itemconfig(self.score_text1, text=str(self.score1))
        self.canvas.itemconfig(self.score_text2, text=str(self.score2))

    def update_ball_position(self) -> None:
        """Update ball position and handle collisions with walls and paddles."""
        # Update ball position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # Handle top and bottom wall collisions
        if self.ball_pos[1] <= BALL_RADIUS:
            self.ball_pos[1] = BALL_RADIUS
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball_pos[1] >= HEIGHT - BALL_RADIUS:
            self.ball_pos[1] = HEIGHT - BALL_RADIUS
            self.ball_vel[1] = -self.ball_vel[1]

        # Handle paddle collisions and scoring
        if self.ball_pos[0] <= BALL_RADIUS + PAD_WIDTH:
            if (
                self.paddle1_pos - HALF_PAD_HEIGHT
                <= self.ball_pos[1]
                <= self.paddle1_pos + HALF_PAD_HEIGHT
            ):
                # Reflect off left paddle
                self.ball_pos[0] = BALL_RADIUS + PAD_WIDTH
                self.ball_vel[0] = -self.ball_vel[0]
                # Add small random angle for variability
                self.ball_vel[1] += random.uniform(-0.5, 0.5)
            else:
                # Player 2 scores
                self.score2 += 1
                self.update_score_display()
                self.spawn_ball(RIGHT)

        elif self.ball_pos[0] >= WIDTH - BALL_RADIUS - PAD_WIDTH:
            if (
                self.paddle2_pos - HALF_PAD_HEIGHT
                <= self.ball_pos[1]
                <= self.paddle2_pos + HALF_PAD_HEIGHT
            ):
                # Reflect off right paddle
                self.ball_pos[0] = WIDTH - BALL_RADIUS - PAD_WIDTH
                self.ball_vel[0] = -self.ball_vel[0]
                # Add small random angle for variability
                self.ball_vel[1] += random.uniform(-0.5, 0.5)
            else:
                # Player 1 scores
                self.score1 += 1
                self.update_score_display()
                self.spawn_ball(LEFT)

    def update_paddle_position(self) -> None:
        """Update paddle positions based on velocity and constrain to canvas bounds."""
        # Update positions
        self.paddle1_pos += self.paddle1_vel
        self.paddle2_pos += self.paddle2_vel

        # Constrain paddles to stay within canvas
        max_pos = HEIGHT / 2 - HALF_PAD_HEIGHT
        min_pos = -max_pos

        self.paddle1_pos = max(min_pos, min(self.paddle1_pos, max_pos))
        self.paddle2_pos = max(min_pos, min(self.paddle2_pos, max_pos))

    def redraw_canvas(self) -> None:
        """Redraw all game elements on canvas."""
        self.canvas.delete("all")

        # Draw game boundaries
        self.canvas.create_line(WIDTH / 2, 0, WIDTH / 2, HEIGHT, fill="white", width=1)
        self.canvas.create_line(PAD_WIDTH, 0, PAD_WIDTH, HEIGHT, fill="white", width=1)
        self.canvas.create_line(
            WIDTH - PAD_WIDTH, 0, WIDTH - PAD_WIDTH, HEIGHT, fill="white", width=1
        )

        # Draw ball
        self.canvas.create_oval(
            self.ball_pos[0] - BALL_RADIUS,
            self.ball_pos[1] - BALL_RADIUS,
            self.ball_pos[0] + BALL_RADIUS,
            self.ball_pos[1] + BALL_RADIUS,
            fill="white",
        )

        # Draw paddles
        self.canvas.create_line(
            PAD_WIDTH / 2,
            self.paddle1_pos + HEIGHT / 2 - HALF_PAD_HEIGHT,
            PAD_WIDTH / 2,
            self.paddle1_pos + HEIGHT / 2 + HALF_PAD_HEIGHT,
            width=PAD_WIDTH,
            fill="white",
        )

        self.canvas.create_line(
            WIDTH - PAD_WIDTH / 2,
            self.paddle2_pos + HEIGHT / 2 - HALF_PAD_HEIGHT,
            WIDTH - PAD_WIDTH / 2,
            self.paddle2_pos + HEIGHT / 2 + HALF_PAD_HEIGHT,
            width=PAD_WIDTH,
            fill="white",
        )

        # Draw scores
        self.canvas.create_text(
            250, 30, text=str(self.score1), fill="white", font=("Arial", 40)
        )
        self.canvas.create_text(
            350, 30, text=str(self.score2), fill="white", font=("Arial", 40)
        )

    def handle_key_press(self, event: tk.Event) -> None:
        """Handle key press events for paddle movement."""
        key = event.keysym
        if key == "Down":
            self.paddle1_vel = 2.0
        elif key == "Up":
            self.paddle1_vel = -2.0
        elif key == "w":
            self.paddle2_vel = -2.0
        elif key == "s":
            self.paddle2_vel = 2.0

    def handle_key_release(self, event: tk.Event) -> None:
        """Handle key release events to stop paddle movement."""
        key = event.keysym
        if key == "Down" or key == "Up":
            self.paddle1_vel = 0.0
        elif key == "w" or key == "s":
            self.paddle2_vel = 0.0

    def game_loop(self) -> None:
        """Main game loop for updating game state and rendering."""
        self.update_ball_position()
        self.update_paddle_position()
        self.redraw_canvas()
        self.root.after(16, self.game_loop)  # ~60 FPS


if __name__ == "__main__":
    root = tk.Tk()
    game = PongGame(root)
    root.mainloop()
