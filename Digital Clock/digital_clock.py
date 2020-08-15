# use Tkinter to show a digital clock
import time
from tkinter import *

root = Tk()

root.title("Digital Clock")
root.geometry("250x100+0+0")
root.resizable(0,0)

label = Label(root, font=("Arial", 30, 'bold'), bg="blue", fg="powder blue", bd =30)
label.grid(row =0, column=1)

def dig_clock():
    
    text_input = time.strftime("%H:%M:%S") # get the current local time from the PC
    
    label.config(text=text_input)
    
    # calls itself every 200 milliseconds
    # to update the time display as needed
    # could use >200 ms, but display gets jerky
    
    label.after(200, clack)

dig_clock()

root.mainloop()
