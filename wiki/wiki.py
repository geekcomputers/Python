# In this program you ask it about any topic and it will show you the data from wikipedia
# pip install wikipedia

import wikipedia
import tkinter as tk
from tkinter import Label, Button, Entry, Text, messagebox, SOLID, GROOVE, StringVar, WORD, END
#import PIL as ImageTK
from tkinter import messagebox


class main():
    def __init__(self, root):
        self.root = root

        self.root.title("WIKIPEDIA SEARCH")
        self.root.geometry("1920x1080")

        self.lbl1 = Label(
                root,
                text="WIKIPEDIA SUMMARY",
                font=("Verdana", 25, "bold"),
                width=50,
                bg="yellow",
                fg="red",
                relief=SOLID,
        )
        self.lbl1.pack(padx=10, pady=15)

        self.question = StringVar()

        self.quesbox = Entry(
            root,
            text="TELL ME YOUR QUESTION",
            font=("Verdana", 20, "italic"),
            width=80,
            textvariable=self.question,
            relief=GROOVE,
            bd=10,
        )
        self.quesbox.pack()

        self.searchbtn = Button(
            root,
            text="SEARCH",
            font=("Callibri", 18, "bold"),
            width=30,
            relief=GROOVE,
            bg="#4cd137",
            bd=3,
            command=lambda:self.summary("None"),
        )
        self.searchbtn.pack()

        self.answer = Text(
            root,
            height=100,
            width=160,
            font=("Arial", 14),
            wrap=WORD,
            bg="#7CEBC6",
            fg="black",
        )

        self.root.bind("<Return>", self.summary)

    def summary(self, event):
        self.searchbtn["text"] = "Searching..."
        try:
            self.query = wikipedia.page(self.question.get(), auto_suggest=True)
            self.quesbox.delete(0, 'end')
            self.answer.delete('1.0', END)
            self.answer.insert(END, (self.query.summary))

            self.answer.pack()
        except Exception as e:
            error_msg = f"{e}"
            messagebox.showerror("Error", error_msg)

        self.searchbtn["text"] = "Search"


        # Wikipeida page returns to many pages

if __name__ == "__main__":
    root = tk.Tk()
    main(root)
    root.mainloop()
