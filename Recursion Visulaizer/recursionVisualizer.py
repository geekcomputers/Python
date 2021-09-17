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
        tree(3*i/4)
        t.right(20)
        tree(3*i/4)
        t.left(10)
        t.right(5)
        t.left(2.5)
        t.backward(i)

tree(40)
tree(25)
tree(30)
tree(46)
tree(38)
tree(76)
tree(90)
tree(35)
tree(40)
tree(18)
tree(36)
tree(5)
tree(10)
tree(15)
tree(30)
turtle.done()
