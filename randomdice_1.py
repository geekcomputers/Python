# GGearing 01/10/19
# Random Dice Game using Tkinter
# Tkinter is used for Making Using GUI in Python Program!
# randint provides you with a random number within your given range!
from random import randint
from tkinter import *

# Function to rool the dice
def roll():
    text.delete(0.0, END)
    text.insert(END, str(randint(1, 100)))


# Defining our GUI
window = Tk()
text = Text(window, width=3, height=1)
buttonA = Button(window, text="Press to roll!", command=roll)
text.pack()
buttonA.pack()
# End Of The Program!
