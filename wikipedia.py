import wikipedia
from tkinter import ttk
from tkinter import *
from tkinter.messagebox import showinfo

win = Tk() #objek
win['background']='gray11'
win.title('Wikipedia')

#function
def search_wiki() :
    search = entry.get()
    Hasil = wikipedia.summary(search)
    showinfo("Hasil Pencarian",Hasil)

label = Label(win,text="Wikipedia Search :",bg='gold',font=('georgia'),fg='blue')
label.grid(row=0,column=0)

entry = ttk.Entry(win,font=('georgia'))
entry.grid(row=0,column=1)

button = ttk.Button(win,text="Search",command=search_wiki)
button.grid(row=0,column=2,padx=10)

win.mainloop()
