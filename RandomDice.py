# GGearing 01/10/19
# Random Dice Game using Tkinter

from random import randint
from tkinter import *


def roll():
    text.delete(0.0, END)
    text.insert(END, str(randint(1, 100)))


window = Tk()
text = Text(window, width=3, height=1)
buttonA = Button(window, text="Press to roll!", command=roll)
text.pack()
buttonA.pack()
