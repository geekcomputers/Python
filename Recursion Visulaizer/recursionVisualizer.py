import turtle
import random
t = turtle.Turtle()
t.right(90)
t.speed(100)
t.left(90)

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
        print('tree execution complete')

def cycle(i):
    if i<100:
        return 
    else:
        try:
            tree(i)
        except:
            print('An exception occured')
        else:
            print('No Exception occured')   
        print('cycle loop complete')

def fractal(i):
    if i<100:
        return
    else:
         cycle(random.randint(1,i+1))
         cycle(random.randint(1,i))
         cycle(random.randint(1,i-1))
         cycle(random.randint(1,i-2))
         print('fractal execution complete')

fractal(random.randint(1,20000))
print('Execution complete')
turtle.done()
