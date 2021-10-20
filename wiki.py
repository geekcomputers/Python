
# Made by abhra kanti Dubey
#In this program you ask it about any topic and it will show you the data from wikipedia
#pip install wikipedia

import wikipedia
import tkinter as tk
from tkinter import *
import PIL as ImageTK
from tkinter import messagebox


root=tk.Tk()
root.title("WIKIPEDIA SEARCH")
root.geometry("1920x1080")


def summary():
    query= wikipedia.page(question.get())
    answer=Text(root,height=100,width=160,font=("Arial",14),wrap=WORD,bg="#7CEBC6" ,fg="black")
    answer.insert(END,(query.summary))
    answer.pack()
    



lbl1= Label(
	root,
	text="WIKIPEDIA SUMMARY TELLER BY ABHRA ",
	font=("Verdana",25,"bold"),
	width=50,
	bg="yellow",
	fg="red",
	relief=SOLID
	)
lbl1.pack(padx=10,pady=15)

question=StringVar()

quesbox=Entry(
	root,
	text='TELL ME YOUR QUESTION',
	font=("Verdana",20,"italic"),
	width=80,
	textvariable=question,
	relief=GROOVE,
	bd=10).pack()

searchbtn=Button(
	root,
	text="SEARCH",
	font=("Callibri",18,"bold"),
	width=30,
	relief=GROOVE,
	bg="#4cd137",
	bd=3,
                  command=summary,).pack()


root.mainloop()
