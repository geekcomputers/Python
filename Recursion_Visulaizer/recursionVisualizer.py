import turtle
import random

t = turtle.Turtle()
num = random.randint(1, 1000)
t.right(num)
t.speed(num)
t.left(num)


def tree(i):
    if i < 10:
        return
    else:
        t.right(15)
        t.forward(15)
        t.left(20)
        t.backward(20)
        tree(2 * i / 5)
        t.left(2)
        tree(3 * i / 4)
        t.left(2)
        tree(i / 2)
        t.backward(num / 5)
        tree(random.randint(1, 100))
        tree(random.randint(1, num))
        tree(random.randint(1, num / 2))
        tree(random.randint(1, num / 3))
        tree(random.randint(1, num / 2))
        tree(random.randint(1, num))
        tree(random.randint(1, 100))
        t.forward(num / 5)
        t.right(2)
        tree(3 * i / 4)
        t.right(2)
        tree(2 * i / 5)
        t.right(2)
        t.left(10)
        t.backward(10)
        t.right(15)
        t.forward(15)
        print("tree execution complete")


def cycle(i):
    if i < 10:
        return
    else:
        try:
            tree(random.randint(1, i))
            tree(random.randint(1, i * 2))
        except:
            print("An exception occured")
        else:
            print("No Exception occured")
        print("cycle loop complete")


def fractal(i):
    if i < 10:
        return
    else:
        cycle(random.randint(1, i + 1))
        cycle(random.randint(1, i))
        cycle(random.randint(1, i - 1))
        cycle(random.randint(1, i - 2))
        print("fractal execution complete")


fractal(random.randint(1, 200))
print("Execution complete")
turtle.done()
