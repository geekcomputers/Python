import turtle

class ShapeDrawer:
    def __init__(self, color, pensize):
        self.turtle = turtle.Turtle()
        self.turtle.color(color)
        self.turtle.pensize(pensize)

    def draw_rectangle(self, width, height):
        for _ in range(2):
            self.turtle.forward(width)
            self.turtle.left(90)
            self.turtle.forward(height)
            self.turtle.left(90)

    def draw_triangle(self, length):
        for _ in range(3):
            self.turtle.forward(length)
            self.turtle.left(120)

def main():
    scrn = turtle.Screen()
    scrn.bgcolor("lavender")

    # Draw Rectangle
    rectangle_drawer = ShapeDrawer("blue", 3)
    rectangle_drawer.draw_rectangle(180, 75)

    # Draw Triangle
    triangle_drawer = ShapeDrawer("hot pink", 4)
    triangle_drawer.turtle.penup()
    triangle_drawer.turtle.goto(-90, -75)
    triangle_drawer.turtle.pendown()
    triangle_drawer.draw_triangle(100)

    # Add more drawings as needed
    # ...

    # Example: Draw a circle
    circle_drawer = ShapeDrawer("green", 2)
    circle_drawer.turtle.penup()
    circle_drawer.turtle.goto(0, 0)
    circle_drawer.turtle.pendown()
    circle_drawer.turtle.circle(50)

    scrn.exitonclick()

if __name__ == "__main__":
    main()