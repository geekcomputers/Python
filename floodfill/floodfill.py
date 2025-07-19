import sys
from collections import deque

import pygame


class FloodFill:
    def __init__(self, window_width: int = 800, window_height: int = 600):
        """Initialize the FloodFill visualization with specified window dimensions."""
        self.window_width = window_width
        self.window_height = window_height
        
        pygame.init()
        pygame.display.set_caption("Floodfill Visualization")
        self.display = pygame.display.set_mode((window_width, window_height))
        self.surface = pygame.Surface(self.display.get_size())
        self.surface.fill((0, 0, 0))
        
        self.generate_closed_polygons()
        self.queue = deque()  # Use deque for O(1) append/pop operations
        self.white = self.surface.map_rgb((255, 255, 255))
        self.black = self.surface.map_rgb((0, 0, 0))

    def generate_closed_polygons(self) -> None:
        """Generate random closed polygons for visualization."""
        if self.window_height < 128 or self.window_width < 128:
            return  # Surface too small
        
        from math import cos, pi, sin
        from random import randint, uniform

        for _ in range(randint(0, 5)):
            x = randint(50, self.window_width - 50)
            y = randint(50, self.window_height - 50)
            angle = uniform(0, 0.7)
            vertices = []

            for _ in range(randint(3, 7)):
                dist = randint(10, 50)
                vertices.append((int(x + cos(angle) * dist), int(y + sin(angle) * dist)))
                angle += uniform(0, pi / 2)

            # Draw polygon edges
            for i in range(len(vertices) - 1):
                pygame.draw.line(self.surface, (255, 0, 0), vertices[i], vertices[i + 1])
            pygame.draw.line(self.surface, (255, 0, 0), vertices[-1], vertices[0])

    def run(self) -> None:
        """Main loop for the visualization."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            events = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    events.append(event)
            
            self.update(events)
            self.display.blit(self.surface, (0, 0))
            pygame.display.flip()
            clock.tick(60)  # Limit FPS for performance
        
        pygame.quit()

    def update(self, events: list) -> None:
        """Handle events and update floodfill algorithm."""
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.queue.append(event.pos)

        if not self.queue:
            return

        # Process multiple points per frame for faster filling
        for _ in range(min(100, len(self.queue))):
            x, y = self.queue.popleft()
            
            # Check bounds and color
            if not (0 <= x < self.window_width and 0 <= y < self.window_height):
                continue
                
            with pygame.PixelArray(self.surface) as pixels:
                if pixels[x, y] != self.black:
                    continue
                    
                pixels[x, y] = (255, 255, 255)
                
                # Check and add neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.window_width and 0 <= ny < self.window_height 
                        and pixels[nx, ny] == self.black):
                        self.queue.append((nx, ny))

if __name__ == "__main__":
    # Set default dimensions
    DEFAULT_WIDTH = 800
    DEFAULT_HEIGHT = 600
    
    # Parse command-line arguments with defaults
    try:
        width = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WIDTH
        height = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_HEIGHT
    except ValueError:
        print("Error: Width and height must be integers.")
        sys.exit(1)
    
    # Validate dimensions
    if width <= 0 or height <= 0:
        print("Error: Width and height must be positive integers.")
        sys.exit(1)
    
    # Initialize and run visualization
    floodfill = FloodFill(width, height)
    floodfill.run()