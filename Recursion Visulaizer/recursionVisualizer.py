import turtle
t = turtle.Turtle()
t.left(90)
t.speed(100)

def tree(i):
    if i<10:
        return
    else:
        t.forward(i)
        t.left(10)
        t.right(5)
        t.left(2.5)
        t.backward(20)
        tree(3*i/4)
        t.right(20)
        tree(3*i/4)
        t.left(10)
        t.right(5)
        t.left(2.5)
        t.forward(15)
        t.backward(i)
tree(40)
tree(30)
tree(40)
turtle.done()
