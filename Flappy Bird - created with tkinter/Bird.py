from collections.abc import Callable
from threading import Thread

from PIL.Image import Image
from PIL.Image import open as openImage
from PIL.ImageTk import PhotoImage


class Background:
    """Mock Background class for type checking"""

    def __init__(self):
        self.bird_image = None

    def create_image(self, x: int, y: int, image: PhotoImage, tag: str) -> int:
        return 0

    def bbox(self, tag: str) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)

    def find_overlapping(self, x1: int, y1: int, x2: int, y2: int) -> list[int]:
        return []

    def move(self, tag: str, x: int, y: int) -> None:
        pass

    def after(self, delay: int, callback: Callable) -> None:
        pass

    def bind(self, event: str, callback: Callable) -> None:
        pass

    def focus_force(self) -> None:
        pass

    def getBackgroundID(self) -> list[int]:
        return []


class Bird(Thread):
    """
    Class for creating a bird object that can fly in a game environment.
    """

    __tag: str = "Bird"
    __isAlive: bool = None
    __going_up: bool = False
    __going_down: float = 0
    __times_skipped: int = 0
    __running: bool = False

    decends: float = 0.00390625
    climbsUp: float = 0.0911458333

    def __init__(
        self,
        background: Background,
        gameover_function: Callable[[], None],
        *screen_geometry: int,
        fp: str = "bird.png",
        event: str = "<Up>",
        descend_speed: int = 5,
    ) -> None:
        """
        Initialize the bird object.

        Args:
            background: The game background canvas.
            gameover_function: Function to call when the bird dies.
            screen_geometry: Width and height of the game screen.
            fp: Path to the bird image file.
            event: Keyboard event to trigger the bird's jump.
            descend_speed: Speed at which the bird descends.
        """
        # Validate input types
        if not isinstance(background, Background):
            raise TypeError(
                "The background argument must be an instance of Background."
            )
        if not callable(gameover_function):
            raise TypeError("The gameover_function argument must be a callable object.")

        # Instance parameters
        self.__canvas: Background = background
        self.image_path: str = fp
        self.__descend_speed: int = descend_speed
        self.gameover_method: Callable[[], None] = gameover_function

        # Get screen dimensions
        if len(screen_geometry) != 2:
            raise ValueError("screen_geometry must contain width and height")
        self.__width: int = screen_geometry[0]
        self.__height: int = screen_geometry[1]

        # Adjust bird movement speeds based on screen height
        self.decends *= self.__height
        self.decends = int(self.decends + 0.5)
        self.climbsUp *= self.__height
        self.climbsUp = int(self.climbsUp + 0.5)

        # Initialize Thread
        super().__init__()

        # Calculate bird dimensions
        self.width: int = (self.__width // 100) * 6
        self.height: int = (self.__height // 100) * 11

        # Load and create bird image
        self.__canvas.bird_image, _, _ = self.getPhotoImage(
            image_path=self.image_path,
            width=self.width,
            height=self.height,
            closeAfter=True,
        )
        self.__birdID: int = self.__canvas.create_image(
            self.__width // 2,
            self.__height // 2,
            image=self.__canvas.bird_image,
            tag=self.__tag,
        )

        # Bind jump event
        self.__canvas.focus_force()
        self.__canvas.bind(event, self.jumps)
        self.__isAlive = True

    def birdIsAlive(self) -> bool:
        """Check if the bird is alive."""
        return self.__isAlive

    def checkCollision(self) -> bool:
        """
        Check if the bird has collided with the boundaries or other objects.

        Returns:
            True if a collision occurred, False otherwise.
        """
        # Get bird position
        position: list[int] = list(self.__canvas.bbox(self.__tag))

        # Check boundary collisions
        if position[3] >= self.__height + 20:
            self.__isAlive = False

        if position[1] <= -20:
            self.__isAlive = False

        # Adjust collision boundaries with a margin of error
        position[0] += int(25 / 78 * self.width)
        position[1] += int(25 / 77 * self.height)
        position[2] -= int(20 / 78 * self.width)
        position[3] -= int(10 / 77 * self.width)

        # Define objects to ignore in collisions
        ignored_collisions: list[int] = self.__canvas.getBackgroundID()
        ignored_collisions.append(self.__birdID)

        # Check for overlapping objects
        possible_collisions: list[int] = list(self.__canvas.find_overlapping(*position))

        # Remove ignored objects from collision list
        for obj_id in ignored_collisions:
            try:
                possible_collisions.remove(obj_id)
            except ValueError:
                continue

        # Collision detected
        if len(possible_collisions) >= 1:
            self.__isAlive = False

        return not self.__isAlive

    def getTag(self) -> str:
        """Return the bird's tag."""
        return self.__tag

    @staticmethod
    def getPhotoImage(
        image: Image | None = None,
        image_path: str | None = None,
        width: int | None = None,
        height: int | None = None,
        closeAfter: bool = False,
    ) -> tuple[PhotoImage, Image | None, Image | None]:
        """
        Create a PhotoImage from a PIL Image or image path.

        Args:
            image: PIL Image object.
            image_path: Path to an image file.
            width: Desired width of the image.
            height: Desired height of the image.
            closeAfter: Whether to close the image after creating the PhotoImage.

        Returns:
            A tuple containing the PhotoImage, the resized image, and the original image.
        """
        if image is None:
            if image_path is None:
                raise ValueError("Either image or image_path must be provided")

            # Open image from path
            image = openImage(image_path)

        # Resize image if dimensions are provided
        if width is None:
            width = image.width
        if height is None:
            height = image.height

        # Create resized image
        newImage: Image = image.resize([width, height])

        # Create PhotoImage
        photoImage: PhotoImage = PhotoImage(newImage)

        # Close images if requested
        if closeAfter:
            newImage.close()
            newImage = None
            image.close()
            image = None

        return photoImage, newImage, image

    def jumps(self, event: object | None = None) -> None:
        """
        Make the bird jump.

        Args:
            event: Keyboard event (automatically passed by Tkinter).
        """
        # Check collision status
        self.checkCollision()

        # Prevent jumping if bird is dead or not running
        if not self.__isAlive or not self.__running:
            self.__going_up = False
            return

        # Bird is going up
        self.__going_up = True
        self.__going_down = 0

        # Animate the jump
        if self.__times_skipped < self.climbsUp:
            # Move bird upwards
            self.__canvas.move(self.__tag, 0, -1)
            self.__times_skipped += 1

            # Continue jump animation
            self.__canvas.after(3, self.jumps)
        else:
            # Jump animation complete
            self.__going_up = False
            self.__times_skipped = 0

    def kill(self) -> None:
        """Kill the bird (set isAlive to False)."""
        self.__isAlive = False

    def run(self) -> None:
        """Main animation loop for the bird's falling motion."""
        self.__running = True

        # Check collision status
        self.checkCollision()

        # Increase falling speed up to maximum
        if self.__going_down < self.decends:
            self.__going_down += 0.05

        # Continue animation if bird is alive
        if self.__isAlive:
            # Only move down if not jumping
            if not self.__going_up:
                self.__canvas.move(self.__tag, 0, self.__going_down)

            # Schedule next frame
            self.__canvas.after(self.__descend_speed, self.run)
        else:
            # Bird is dead, trigger game over
            self.__running = False
            self.gameover_method()
