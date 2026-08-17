from tkinter import *
from translate import Translator

# Translator
def translate_text():
    try:
        translator = Translator(from_lang=lan1.get(), to_lang=lan2.get())
        translation = translator.translate(text_input.get())
        output_text.set(translation)
    except Exception as e:
        output_text.set("Error")

# root window
root = Tk()
root.title("Translator")

# frame
mainframe = Frame(root)
mainframe.pack(pady=50, padx=50)

lan1 = StringVar(value="en")   # default: English
lan2 = StringVar(value="hi")   # default: Hindi
text_input = StringVar()
output_text = StringVar()

# input fields
Label(mainframe, text="From (e.g. en)").grid(row=0, column=0)
Entry(mainframe, textvariable=lan1).grid(row=1, column=0, padx=10, pady=10)

Label(mainframe, text="To (e.g. hi)").grid(row=0, column=1)
Entry(mainframe, textvariable=lan2).grid(row=1, column=1, padx=10, pady=10)

# Text input
Label(mainframe, text="Enter text").grid(row=2, column=0)
Entry(mainframe, textvariable=text_input).grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Output
Label(mainframe, text="Output").grid(row=4, column=0)
Entry(mainframe, textvariable=output_text).grid(row=5, column=0, columnspan=2, padx=10, pady=10)

Button(mainframe, text="Translate", command=translate_text, bg="green").grid(row=6, column=0, columnspan=2)

root.mainloop()
