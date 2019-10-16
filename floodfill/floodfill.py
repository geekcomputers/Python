import pygame

'''
Visualises how a floodfill algorithm runs and work using pygame
Pass two int arguments for the window width and the window height
`python floodfill.py <width> <height>`
'''

class FloodFill:
    def __init__(self, window_width, window_height):
        self.window_width = int(window_width)
        self.window_height = int(window_height)

        pygame.init()
        pygame.display.set_caption("Floodfill")
        self.display = pygame.display.set_mode((self.window_width, self.window_height))
        self.surface = pygame.Surface(self.display.get_size())
        self.surface.fill((0, 0, 0))

        self.generateClosedPolygons() # for visualisation purposes

        self.queue = []

    def generateClosedPolygons(self):
        if self.window_height < 128 or self.window_width < 128:
            return # surface too small

        from random import randint, uniform
        from math import pi, sin, cos
        for n in range(0, randint(0, 5)):
            x = randint(50, self.window_width - 50)
            y = randint(50, self.window_height - 50)

            angle = 0
            angle += uniform(0, 0.7)
            vertices = []

            for i in range(0, randint(3, 7)):
                dist = randint(10, 50)
                vertices.append((int(x + cos(angle) * dist), int(y + sin(angle) * dist)))
                angle += uniform(0, pi/2)

            for i in range(0, len(vertices) - 1):
                pygame.draw.line(self.surface, (255, 0, 0), vertices[i], vertices[i + 1])

            pygame.draw.line(self.surface, (255, 0, 0), vertices[len(vertices) - 1], vertices[0])

    def run(self):
        looping = True
        while looping:
            evsforturn = []
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    looping = False
                else:
                    evsforturn.append(ev) # TODO: Maybe extend with more events
            self.update(evsforturn)
            self.display.blit(self.surface, (0, 0))
            pygame.display.flip()

        pygame.quit()

    def update(self, events):
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self.queue.append(ev.pos)

        if not len(self.queue):
            return

        point = self.queue.pop(0)

        pixArr = pygame.PixelArray(self.surface)

        if pixArr[point[0], point[1]] == self.surface.map_rgb((255, 255, 255)):
            return

        pixArr[point[0], point[1]] = (255, 255, 255)

        left = (point[0] - 1, point[1])
        right = (point[0] + 1, point[1])
        top = (point[0], point[1] + 1)
        bottom = (point[0], point[1] - 1)

        if self.inBounds(left) and left not in self.queue and pixArr[left[0], left[1]] == self.surface.map_rgb((0, 0, 0)):
            self.queue.append(left)
        if self.inBounds(right) and right not in self.queue and pixArr[right[0], right[1]] == self.surface.map_rgb((0, 0, 0)):
            self.queue.append(right)
        if self.inBounds(top) and top not in self.queue and pixArr[top[0], top[1]] == self.surface.map_rgb((0, 0, 0)):
            self.queue.append(top)
        if self.inBounds(bottom) and bottom not in self.queue and pixArr[bottom[0], bottom[1]] == self.surface.map_rgb((0, 0, 0)):
            self.queue.append(bottom)
        
        del pixArr

    def inBounds(self, coord):
        if coord[0] < 0 or coord[0] >= self.window_width:
            return False
        elif coord[1] < 0 or coord[1] >= self.window_height:
            return False
        return True

if __name__ == "__main__":
    import sys
    floodfill = FloodFill(sys.argv[1], sys.argv[2])
    floodfill.run()