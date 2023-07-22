# Emoji Dictionary

# -----------------------------------------------------------------------------------------------------
import io  # used for dealing with input and output
from tkinter import *  # importing the necessary libraries
import tkinter.messagebox as mbox
import tkinter as tk  # imported tkinter as tk
import emoji

# -----------------------------------------------------------------------------------------------


class Keypad(tk.Frame):

    cells = [
        ["😀", "🥰", "😴", "🤓", "🤮", "🤬", "😨", "🤑", "😫", "😎"],
        [
            "🐒",
            "🐕",
            "🐎",
            "🐪",
            "🐁",
            "🐘",
            "🦘",
            "🦈",
            "🐓",
            "🐝",
            "👀",
            "🦴",
            "👩🏿",
            "‍🤝",
            "🧑",
            "🏾",
            "👱🏽",
            "‍♀",
            "🎞",
            "🎨",
            "⚽",
        ],
        [
            "🍕",
            "🍗",
            "🍜",
            "☕",
            "🍴",
            "🍉",
            "🍓",
            "🌴",
            "🌵",
            "🛺",
            "🚲",
            "🛴",
            "🚉",
            "🚀",
            "✈",
            "🛰",
            "🚦",
            "🏳",
            "‍🌈",
            "🌎",
            "🧭",
        ],
        [
            "🔥",
            "❄",
            "🌟",
            "🌞",
            "🌛",
            "🌝",
            "🌧",
            "🧺",
            "🧷",
            "🪒",
            "⛲",
            "🗼",
            "🕌",
            "👁",
            "‍🗨",
            "💬",
            "™",
            "💯",
            "🔕",
            "💥",
            "❤",
        ],
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target = None
        self.memory = ""

        for y, row in enumerate(self.cells):
            for x, item in enumerate(row):
                b = tk.Button(
                    self,
                    text=item,
                    command=lambda text=item: self.append(text),
                    font=("Arial", 14),
                    bg="yellow",
                    fg="blue",
                    borderwidth=3,
                    relief="raised",
                )
                b.grid(row=y, column=x, sticky="news")

        x = tk.Button(
            self,
            text="Space",
            command=self.space,
            font=("Arial", 14),
            bg="yellow",
            fg="blue",
            borderwidth=3,
            relief="raised",
        )
        x.grid(row=0, column=10, columnspan="2", sticky="news")

        x = tk.Button(
            self,
            text="tab",
            command=self.tab,
            font=("Arial", 14),
            bg="yellow",
            fg="blue",
            borderwidth=3,
            relief="raised",
        )
        x.grid(row=0, column=12, columnspan="2", sticky="news")

        x = tk.Button(
            self,
            text="Backspace",
            command=self.backspace,
            font=("Arial", 14),
            bg="yellow",
            fg="blue",
            borderwidth=3,
            relief="raised",
        )
        x.grid(row=0, column=14, columnspan="3", sticky="news")

        x = tk.Button(
            self,
            text="Clear",
            command=self.clear,
            font=("Arial", 14),
            bg="yellow",
            fg="blue",
            borderwidth=3,
            relief="raised",
        )
        x.grid(row=0, column=17, columnspan="2", sticky="news")

        x = tk.Button(
            self,
            text="Hide",
            command=self.hide,
            font=("Arial", 14),
            bg="yellow",
            fg="blue",
            borderwidth=3,
            relief="raised",
        )
        x.grid(row=0, column=19, columnspan="2", sticky="news")

    def get(self):
        if self.target:
            return self.target.get()

    def append(self, text):
        if self.target:
            self.target.insert("end", text)

    def clear(self):
        if self.target:
            self.target.delete(0, END)

    def backspace(self):
        if self.target:
            text = self.get()
            text = text[:-1]
            self.clear()
            self.append(text)

    def space(self):
        if self.target:
            text = self.get()
            text = text + " "
            self.clear()
            self.append(text)

    def tab(self):  # 5 spaces
        if self.target:
            text = self.get()
            text = text + "     "
            self.clear()
            self.append(text)

    def copy(self):
        # TODO: copy to clipboad
        if self.target:
            self.memory = self.get()
            self.label["text"] = "memory: " + self.memory
            print(self.memory)

    def paste(self):
        # TODO: copy from clipboad
        if self.target:
            self.append(self.memory)

    def show(self, entry):
        self.target = entry

        self.place(relx=0.5, rely=0.6, anchor="c")

    def hide(self):
        self.target = None

        self.place_forget()


# function defined th=o clear both the input text and output text --------------------------------------------------
def clear_text():
    inputentry.delete(0, END)
    outputtxt.delete("1.0", "end")


# function to search emoji
def search_emoji():
    word = inputentry.get()
    if word == "":
        outputtxt.insert(END, "You have entered no emoji.")
    else:
        means = emoji.demojize(word)
        outputtxt.insert(END, "Meaning of Emoji  :  " + str(word) + "\n\n" + means)


# main window created
window = tk.Tk()
window.title("Emoji Dictionary")
window.geometry("1000x700")

# for writing Dictionary label, at the top of window
dic = tk.Label(
    text="EMOJI DICTIONARY", font=("Arial", 50, "underline"), fg="magenta"
)  # same way bg
dic.place(x=160, y=10)

start1 = tk.Label(
    text="Enter any Emoji you want to search...", font=("Arial", 30), fg="green"
)  # same way bg
start1.place(x=160, y=120)

myname = StringVar(window)
firstclick1 = True


def on_inputentry_click(event):
    """function that gets called whenever entry1 is clicked"""
    global firstclick1

    if firstclick1:  # if this is the first time they clicked it
        firstclick1 = False
        inputentry.delete(0, "end")  # delete all the text in the entry


# Taking input from TextArea
# inputentry = Entry(window,font=("Arial", 35), width=33, border=2)
inputentry = Entry(
    window, font=("Arial", 35), width=28, border=2, bg="light yellow", fg="brown"
)
inputentry.place(x=120, y=180)

# # Creating Search Button
Button(
    window,
    text="🔍 SEARCH",
    command=search_emoji,
    font=("Arial", 20),
    bg="light green",
    fg="blue",
    borderwidth=3,
    relief="raised",
).place(x=270, y=250)

# # creating clear button
Button(
    window,
    text="🧹 CLEAR",
    command=clear_text,
    font=("Arial", 20),
    bg="orange",
    fg="blue",
    borderwidth=3,
    relief="raised",
).place(x=545, y=250)

# meaning label
start1 = tk.Label(text="Meaning...", font=("Arial", 30), fg="green")  # same way bg
start1.place(x=160, y=340)

# # Output TextBox Creation
outputtxt = tk.Text(
    window,
    height=7,
    width=57,
    font=("Arial", 17),
    bg="light yellow",
    fg="brown",
    borderwidth=3,
    relief="solid",
)
outputtxt.place(x=120, y=400)

# function for exiting
def exit_win():
    if mbox.askokcancel("Exit", "Do you want to exit?"):
        window.destroy()


# # creating exit button
Button(
    window,
    text="❌ EXIT",
    command=exit_win,
    font=("Arial", 20),
    bg="red",
    fg="black",
    borderwidth=3,
    relief="raised",
).place(x=435, y=610)

keypad = Keypad(window)

# # creating speech to text button
v_keypadb = Button(
    window,
    text="⌨",
    command=lambda: keypad.show(inputentry),
    font=("Arial", 18),
    bg="light yellow",
    fg="green",
    borderwidth=3,
    relief="raised",
).place(x=870, y=183)

window.protocol("WM_DELETE_WINDOW", exit_win)
window.mainloop()
