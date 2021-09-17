import turtle
t = turtle.Turtle()
t.left(90)
t.speed(200)

def tree(i):
    if i<10:
        return
    else:
        t.forward(i)
        t.left(10)
        t.right(5)
        t.left(2.5)
        t.backward(10)
        tree(3*i/4)
        t.right(20)
        tree(3*i/4)
        t.left(10)
        t.right(5)
        t.left(2.5)
        t.forward(10)
        t.backward(i)

tree(40)
tree(35)
tree(30)
tree(25)
tree(20)
tree(25)
tree(35)
tree(40)
turtle.done()
