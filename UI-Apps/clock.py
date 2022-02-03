import tkinter

# retrieve system's time
from time import strftime

# ------------------main code-----------------------
# initializing the main UI object
top = tkinter.Tk()
# setting title of the App
top.title("Clock")
# restricting the resizable property
top.resizable(0, 0)


def time():
    string = strftime("%H:%M:%S %p")
    clockTime.config(text=string)
    clockTime.after(1000, time)


clockTime = tkinter.Label(
    top, font=("calibri", 40, "bold"), background="black", foreground="white"
)

clockTime.pack(anchor="center")
time()


top.mainloop()
