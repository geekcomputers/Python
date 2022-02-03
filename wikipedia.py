import wikipedia
from tkinter import *
from tkinter.messagebox import showinfo

win = Tk()  # objek
win.title("WIKIPEDIA")
win.geometry("200x70")  # function

# function
def search_wiki():
    search = entry.get()
    Hasil = wikipedia.summary(search)
    showinfo("Hasil Pencarian", Hasil)


label = Label(win, text="Wikipedia Search :")
label.grid(row=0, column=0)

entry = Entry(win)
entry.grid(row=1, column=0)

button = Button(win, text="Search", command=search_wiki)
button.grid(row=1, column=1, padx=10)

win.mainloop()
