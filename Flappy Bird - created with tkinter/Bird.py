from threading import Thread, Lock
from enum import Enum, auto
from Background import Background
from PIL.Image import open as open_image
from PIL.ImageTk import PhotoImage

class BirdState(Enum):
    """
    Enumeration for bird state.
    """
    ALIVE = auto()
    DEAD = auto()
    FLYING = auto()

class Bird(Thread):
    """
    Class to create a bird character with movement and collision detection.
    """

    DEFAULT_TAG = "Bird"
    DESCEND_RATE = 0.00390625
    CLIMB_RATE = 0.0911458333

    def __init__(self, background, gameover_function, *screen_geometry, 
                 fp="bird.png", event="<Up>", descend_speed=5):
        """
        Initialize the Bird object.

        :param background: Background instance where the bird is drawn.
        :param gameover_function: Function to call when the game is over.
        :param screen_geometry: Tuple of screen width and height.
        :param fp: File path to the bird image.
        :param event: Key event for bird jump.
        :param descend_speed: Speed of descending.
        """
        if not isinstance(background, Background):
            raise TypeError("The background argument must be an instance of Background.")
        if not callable(gameover_function):
            raise TypeError("The gameover_function argument must be callable.")
        
        self._canvas = background
        self.image_path = fp
        self._descend_speed = descend_speed
        self._gameover_function = gameover_function

        # Screen dimensions
        self._width, self._height = screen_geometry

        # Set descending and climbing rates
        self.descend_rate = int(self.DESCEND_RATE * self._height + 0.5)
        self.climb_rate = int(self.CLIMB_RATE * self._height + 0.5)

        # Thread initialization
        super().__init__()
        self._lock = Lock()

        # Bird state management
        self.state = BirdState.ALIVE
        self._going_up = False
        self._going_down_speed = 0
        self._jump_skipped = 0
        self._running = False

        # Bird dimensions based on screen size
        self._bird_width = (self._width // 100) * 6
        self._bird_height = (self._height // 100) * 11

        # Load the bird image
        self._load_bird_image()

        # Set up key event for jumping
        self._canvas.focus_force()
        self._canvas.bind(event, self.jump)

    def _load_bird_image(self):
        """
        Load and create the bird image on the canvas.
        """
        try:
            self._canvas.bird_image = self.get_photo_image(
                image_path=self.image_path,
                width=self._bird_width,
                height=self._bird_height,
                close_after=True
            )[0]
            self._bird_id = self._canvas.create_image(
                self._width // 2,
                self._height // 2,
                image=self._canvas.bird_image,
                tag=self.DEFAULT_TAG
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load bird image: {e}")

    @property
    def is_alive(self):
        """
        Property to check if the bird is alive.
        """
        return self.state == BirdState.ALIVE

    def check_collision(self):
        """
        Check if the bird has collided with an object or boundary.
        """
        position = list(self._canvas.bbox(self.DEFAULT_TAG))

        if position[3] >= self._height + 20 or position[1] <= -20:
            self.state = BirdState.DEAD

        margin_x = int(25 / 78 * self._bird_width)
        margin_y = int(25 / 77 * self._bird_height)
        position[0] += margin_x
        position[1] += margin_y
        position[2] -= margin_x
        position[3] -= int(10 / 77 * self._bird_width)

        ignored_ids = self._canvas.getBackgroundID()
        ignored_ids.append(self._bird_id)

        possible_collisions = set(self._canvas.find_overlapping(*position)) - set(ignored_ids)

        if possible_collisions:
            self.state = BirdState.DEAD

        return self.state == BirdState.DEAD

    @staticmethod
    def get_photo_image(image=None, image_path=None, width=None, height=None, close_after=False):
        """
        Returns a PhotoImage object from an image or image path.
        """
        if not image and not image_path:
            return

        image = image or open_image(image_path)

        width = width or image.width
        height = height or image.height

        resized_image = image.resize((width, height))
        photo_image = PhotoImage(resized_image)

        if close_after:
            resized_image.close()
            image.close()

        return photo_image, resized_image, image

    def jump(self, event=None):
        """
        Make the bird jump.
        """
        if not self.is_alive or not self._running:
            return

        self._going_up = True
        self._going_down_speed = 0

        if self._jump_skipped < self.climb_rate:
            self._canvas.move(self.DEFAULT_TAG, 0, -1)
            self._jump_skipped += 1
            self._canvas.after(3, self.jump)
        else:
            self._going_up = False
            self._jump_skipped = 0

    def kill(self):
        """
        Kill the bird, changing its state to DEAD.
        """
        self.state = BirdState.DEAD

    def run(self):
        """
        Run the bird's falling animation.
        """
        self._running = True

        while self.is_alive:
            with self._lock:
                self.check_collision()

                if self._going_down_speed < self.descend_rate:
                    self._going_down_speed += 0.05

                if not self._going_up:
                    self._canvas.move(self.DEFAULT_TAG, 0, self._going_down_speed)

            self._canvas.after(self._descend_speed)

        self._running = False
        self._gameover_function()
