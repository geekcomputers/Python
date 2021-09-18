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
        t.right(10)
        tree(3*i/4)
        t.backward(20)
        tree(2*i/5)
        t.forward(30)
        t.left(40)
        tree(3*i/4)
        print('tree execution complete')

def cycle(i):
    if i<10:
        return 
    else:
        try:
            tree(random.randint(1,i))
            tree(random.randint(1,i*2))
        except:
            print('An exception occured')
        else:
            print('No Exception occured')   
        print('cycle loop complete')

def fractal(i):
    if i<10:
        return
    else:
         cycle(random.randint(1,i+1))
         cycle(random.randint(1,i))
         cycle(random.randint(1,i-1))
         cycle(random.randint(1,i-2))
         print('fractal execution complete')

fractal(random.randint(1,10000))
print('Execution complete')
turtle.done()
