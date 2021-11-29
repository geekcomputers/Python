from tkinter import *
from translate import Translator

# Translator function
def translate():
    translator = Translator(from_lang=lan1.get(), to_lang=lan2.get())
    translation = translator.translate(var.get())
    var1.set(translation)


# Tkinter root Window with title
root = Tk()
root.title("Translator")

# Creating a Frame and Grid to hold the Content
mainframe = Frame(root)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
mainframe.pack(pady=100, padx=100)

# variables
lan1 = StringVar(root)
lan2 = StringVar(root)
lan1.set("English")
lan2.set("Hindi")

# taking input of languages from user
Label(mainframe, text="Enter language translate from").grid(row=0, column=1)
var = StringVar()
textbox = Entry(mainframe, textvariable=var).grid(row=1, column=1, padx=10, pady=10)

Label(mainframe, text="Enter a language to").grid(row=0, column=2)
var = StringVar()
textbox = Entry(mainframe, textvariable=var).grid(row=1, column=2, padx=10, pady=10)

# Text Box to take user input
Label(mainframe, text="Enter text").grid(row=3, column=0)
var = StringVar()
textbox = Entry(mainframe, textvariable=var).grid(row=3, column=1)

# textbox to show output
# label can also be used
Label(mainframe, text="Output").grid(row=3, column=2)
var1 = StringVar()
textbox = Entry(mainframe, textvariable=var1).grid(row=3, column=3, padx=10, pady=10)

# creating a button to call Translator function
b = Button(
    mainframe, text="Translate", command=translate, activebackground="green"
).grid(row=4, column=1, columnspan=3)

root.mainloop()
