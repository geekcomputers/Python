import turtle
import random
t = turtle.Turtle()
t.left(90)
t.speed(100)

def tree(i):
    if i<10:
        return
    else:
        t.forward(i)
        t.backward(i-2)
        t.left(100)
        t.right(50)
        t.left(25)
        t.backward(20)
        t.left(5)
        tree(2*i/5)
        t.right(20)
        tree(2*i/5)
        t.left(10)
        t.right(5)
        t.left(25)
        t.forward(15)
        t.right(50)
        t.left(100)
        t.backward(i-1)
        t.forward(i+2)

def cycle(i):
    if i<100:
        return 
    else:
        tree(i/5)
        tree(i/4)
        tree(i/3)
        tree(i/2)
        tree(i)
        tree(i*1.5)

def fractal(i):
    if i<100:
        return
    else:
         cycle(random.randint(0,i+1))
         cycle(random.randint(0,i))
         cycle(random.randint(0,i-1))
         cycle(random.randint(0,i-2))

fractal(200)
turtle.done()
