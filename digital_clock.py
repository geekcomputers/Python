# master
# importing whole module
# use Tkinter to show a digital clock
# using python code base

import time

# because we need digital clock , so we are importing the time library.
#  master
from tkinter import *
from tkinter.ttk import *

# importing strftime function to
# retrieve system's time
from time import strftime

# creating tkinter window
root = Tk()
root.title("Clock")

# master

# This function is used to
# display time on the label
def def_time():
    string = strftime("%H:%M:%S %p")
    lbl.config(text=string)
    lbl.after(1000, time)


# Styling the label widget so that clock
# will look more attractive
lbl = Label(
    root,
    font=("calibri", 40, "bold", "italic"),
    background="Black",
    foreground="Yellow",
)

# Placing clock at the centre
# of the tkinter window
lbl.pack(anchor="center")
def_time()

mainloop()
# =======
label = Label(root, font=("Arial", 30, "bold"), bg="black", fg="white", bd=30)
label.grid(row=0, column=1)


# function to declare the tkniter clock
def dig_clock():

    text_input = time.strftime("%H : %M : %S")  # get the current local time from the PC

    label.config(text=text_input)

    # calls itself every 200 milliseconds
    # to update the time display as needed
    # could use >200 ms, but display gets jerky

    label.after(200, dig_clock)


# calling the function
dig_clock()

root.mainloop()
# master
