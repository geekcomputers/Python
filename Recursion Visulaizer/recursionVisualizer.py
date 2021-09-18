import turtle
import random
t = turtle.Turtle()
num=random.randint(1,100)
t.right(num)
t.speed(num)
t.left(num)

def tree(i):
    if i<10:
        return
    else:
        t.forward(i)
        t.backward(i-2)
        t.left(i+3)
        t.backward(i-1)
        t.forward(i+2)
        t.right(i+3)
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

fractal(random.randint(1,2000))
print('Execution complete')
turtle.done()
