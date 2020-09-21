import turtle
t= turtle.Turtle()
turtle.title("I Love You")
screen= turtle.Screen()
screen.bgcolor("black")
t.color("red")
t.begin_fill()
t.fillcolor("red")

t.left(140)
t.forward(180)
t.circle(-90, 200)

t.setheading(60)   # t.left
t.circle(-90, 200)
t.forward(180)

t.end_fill()
t.hideturtle()

