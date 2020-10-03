from _datetime import datetime
import tkinter as tk
from tkinter import ttk
from _datetime import *

win = tk.Tk()
win.title('Age Calculate')
win.geometry('310x400')
# win.iconbitmap('pic.png')    this is use extention  ico then show pic 

############################################ Frame ############################################
pic = tk.PhotoImage(file=r"E:\Python Practice\Age_calculate\pic.png")
win.tk.call('wm','iconphoto',win._w,pic)


canvas=tk.Canvas(win,width=310,height=190)
canvas.grid()
image = tk.PhotoImage(file=r"E:\Python Practice\Age_calculate\pic.png")
canvas.create_image(0,0,anchor='nw',image=image)

frame = ttk.Frame(win)
frame.place(x=40,y=220)



############################################ Label on Frame ############################################

name = ttk.Label(frame,text = 'Name : ',font = ('',12,'bold'))
name.grid(row=0,column=0,sticky = tk.W)

year = ttk.Label(frame,text = 'Year : ',font = ('',12,'bold'))
year.grid(row=1,column=0,sticky = tk.W)

month = ttk.Label(frame,text = 'Month : ',font = ('',12,'bold'))
month.grid(row=2,column=0,sticky = tk.W)

date = ttk.Label(frame,text = 'Date : ',font = ('',12,'bold'))
date.grid(row=3,column=0,sticky = tk.W)

############################################ Entry Box ############################################
name_entry = ttk.Entry(frame,width=25)
name_entry.grid(row=0,column=1)
name_entry.focus()

year_entry = ttk.Entry(frame,width=25)
year_entry.grid(row=1,column=1,pady=5)

month_entry = ttk.Entry(frame,width=25)
month_entry.grid(row=2,column=1)

date_entry = ttk.Entry(frame,width=25)
date_entry.grid(row=3,column=1,pady=5)


def age_cal():
    name_entry.get()
    year_entry.get()
    month_entry.get()
    date_entry.get()
    cal = datetime.today()-(int(year_entry))
    print(cal)


btn = ttk.Button(frame,text='Age calculate',command=age_cal)
btn.grid(row=4,column=1)



win.mainloop()
