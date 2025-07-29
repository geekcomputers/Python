import tkinter as tk
from random import shuffle

from freegames import path

# Game configuration
WINDOW_SIZE = 420
GRID_SIZE = 50
GRID_COUNT = 8
TILES_COUNT = GRID_COUNT**2


class MemoryGame:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Memory Match Game")
        self.root.resizable(False, False)

        # Game assets
        self.car_image = tk.PhotoImage(file=path("car.gif"))

        # Game state
        self.tiles = list(range(32)) * 2
        self.state = {"mark": None}
        self.hide = [True] * TILES_COUNT

        # Create canvas
        self.canvas = tk.Canvas(root, width=WINDOW_SIZE, height=WINDOW_SIZE, bg="white")
        self.canvas.pack()

        # Load background image
        self.background = self.canvas.create_image(
            WINDOW_SIZE // 2, WINDOW_SIZE // 2, image=self.car_image
        )

        # Shuffle tiles
        shuffle(self.tiles)

        # Draw initial tiles
        self.draw_tiles()

        # Bind click event
        self.canvas.bind("<Button-1>", self.handle_click)

    def calculate_index(self, x: float, y: float) -> int:
        """Convert (x, y) coordinates to tiles index."""
        grid_x = min(int(x // GRID_SIZE), GRID_COUNT - 1)
        grid_y = min(int(y // GRID_SIZE), GRID_COUNT - 1)
        return grid_y * GRID_COUNT + grid_x

    def calculate_xy(self, count: int) -> tuple[float, float]:
        """Convert tiles count to (x, y) coordinates."""
        x = (count % GRID_COUNT) * GRID_SIZE
        y = (count // GRID_COUNT) * GRID_SIZE
        return (x, y)

    def draw_tiles(self) -> None:
        """Draw all tiles on the canvas."""
        # Clear previous tiles
        self.canvas.delete("tile")

        for count in range(TILES_COUNT):
            if self.hide[count]:
                x, y = self.calculate_xy(count)
                # Draw tile background
                self.canvas.create_rectangle(
                    x,
                    y,
                    x + GRID_SIZE,
                    y + GRID_SIZE,
                    fill="white",
                    outline="black",
                    tags="tile",
                )

        # Show currently selected tile's number
        mark = self.state["mark"]
        if mark is not None and self.hide[mark]:
            x, y = self.calculate_xy(mark)
            self.canvas.create_text(
                x + GRID_SIZE // 2,
                y + GRID_SIZE // 2,
                text=str(self.tiles[mark]),
                font=("Arial", 24, "bold"),
                tags="tile",
            )

        # Check if game is completed
        if all(not h for h in self.hide):
            self.show_completion_message()

    def handle_click(self, event: tk.Event) -> None:
        """Handle mouse click events."""
        x, y = event.x, event.y
        spot = self.calculate_index(x, y)

        # Check if the clicked tile is already revealed
        if not self.hide[spot]:
            return

        current_mark = self.state["mark"]

        # Handle first tile selection or same tile click
        if current_mark is None or current_mark == spot:
            self.state["mark"] = spot
            self.draw_tiles()
            return

        # Handle unmatched tiles
        if self.tiles[current_mark] != self.tiles[spot]:
            self.state["mark"] = spot
            self.draw_tiles()
            return

        # Tiles match, reveal both
        self.hide[spot] = False
        self.hide[current_mark] = False
        self.state["mark"] = None

        # Redraw tiles to reflect changes
        self.draw_tiles()
        self.canvas.after(1000, self.draw_tiles)
        # Check for game completion
        if all(not h for h in self.hide):
            self.show_completion_message()

    def show_completion_message(self) -> None:
        """Show game completion message."""
        self.canvas.delete("all")
        self.canvas.create_image(
            WINDOW_SIZE // 2, WINDOW_SIZE // 2, image=self.car_image
        )
        self.canvas.create_text(
            WINDOW_SIZE // 2,
            WINDOW_SIZE // 2 - 30,
            text="Congratulations!",
            font=("Arial", 32, "bold"),
            fill="white",
        )
        self.canvas.create_text(
            WINDOW_SIZE // 2,
            WINDOW_SIZE // 2 + 30,
            text="You completed the game!",
            font=("Arial", 24, "bold"),
            fill="white",
        )


if __name__ == "__main__":
    root = tk.Tk()
    game = MemoryGame(root)
    root.mainloop()
