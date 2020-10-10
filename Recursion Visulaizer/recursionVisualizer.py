import turtle
t = turtle.Turtle()
t.left(90)
t.speed(200)

def tree(i):
    if i<10:
        return
    else:
        t.forward(i)
        t.left(30)
        tree(3*i/4)
        t.right(60)
        tree(3*i/4)
        t.left(30)
        t.backward(i)

tree(100)
turtle.done()
