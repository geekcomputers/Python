import tkinter

root = tkinter.Tk()
root.geometry("360x470")
root.title("SECRET MESSAGE CODER DECODER")

name1 = tkinter.StringVar()
name2 = tkinter.StringVar()
result1 = tkinter.StringVar()
r1 = tkinter.Label(
    root,
    text="",
    textvariable=result1,
    fg="green",
    bg="white",
    font=("lucida handwriting", 15, "bold", "underline"),
)
r1.place(x=10, y=150)
result2 = tkinter.StringVar()
r2 = tkinter.Label(
    root,
    text="",
    textvariable=result2,
    fg="green",
    bg="white",
    font=("lucida handwriting", 15, "bold", "underline"),
)
r2.place(x=0, y=380)
a = tkinter.Entry(
    root,
    text="",
    textvariable=name1,
    bd=5,
    bg="light grey",
    fg="red",
    font=("bold", 20),
)
a.place(x=0, y=50)
b = tkinter.Entry(
    root,
    text="",
    textvariable=name2,
    bd=5,
    bg="light grey",
    fg="red",
    font=("bold", 20),
)
b.place(x=0, y=270)
t1 = tkinter.Label(
    root, text="TYPE MESSAGE:", font=("arial", 20, "bold", "underline"), fg="red"
)
t2 = tkinter.Label(
    root, text="TYPE SECRET MESSAGE:", font=("arial", 20, "bold", "underline"), fg="red"
)
t1.place(x=10, y=0)
t2.place(x=10, y=220)


def show1():
    data1 = name1.get()
    codes = {
        "b": "a",
        "c": "b",
        "d": "c",
        "e": "d",
        "f": "e",
        "g": "f",
        "h": "g",
        "i": "h",
        "j": "i",
        "k": "j",
        "l": "k",
        "m": "l",
        "n": "m",
        "o": "n",
        "p": "o",
        "q": "p",
        "r": "q",
        "s": "r",
        "t": "s",
        "u": "t",
        "v": "u",
        "w": "v",
        "x": "w",
        "y": "x",
        "z": "y",
        "a": "z",
        " ": " ",
        "B": "A",
        "C": "B",
        "D": "C",
        "E": "D",
        "F": "E",
        "G": "F",
        "H": "G",
        "I": "H",
        "J": "I",
        "K": "J",
        "L": "K",
        "M": "L",
        "N": "M",
        "O": "N",
        "P": "O",
        "Q": "P",
        "R": "Q",
        "S": "R",
        "T": "S",
        "U": "T",
        "V": "U",
        "W": "V",
        "X": "W",
        "Y": "X",
        "Z": "Y",
        "A": "Z",
    }
    lol1 = ""
    for x in data1:
        lol1 = lol1 + codes[x]
    name1.set("")
    result1.set("SECRET MESSAGE IS:-\n" + lol1)
    return


bt1 = tkinter.Button(
    root,
    text="OK",
    bg="white",
    fg="black",
    bd=5,
    command=show1,
    font=("calibri", 15, "bold", "underline"),
)
bt1.place(x=10, y=100)


def show2():
    data2 = name2.get()
    codes = {
        "a": "b",
        "b": "c",
        "c": "d",
        "d": "e",
        "e": "f",
        "f": "g",
        "g": "h",
        "h": "i",
        "i": "j",
        "j": "k",
        "k": "l",
        "l": "m",
        "m": "n",
        "n": "o",
        "o": "p",
        "p": "q",
        "q": "r",
        "r": "s",
        "s": "t",
        "t": "u",
        "u": "v",
        "v": "w",
        "w": "x",
        "x": "y",
        "y": "z",
        "z": "a",
        " ": " ",
        "A": "B",
        "B": "C",
        "C": "D",
        "D": "E",
        "E": "F",
        "F": "G",
        "G": "H",
        "H": "I",
        "I": "J",
        "J": "K",
        "K": "L",
        "L": "M",
        "M": "N",
        "N": "O",
        "O": "P",
        "P": "Q",
        "Q": "R",
        "R": "S",
        "S": "T",
        "T": "U",
        "U": "V",
        "V": "W",
        "W": "X",
        "X": "Y",
        "Y": "Z",
        "Z": "A",
    }
    lol2 = ""
    for x in data2:
        lol2 = lol2 + codes[x]
    name2.set("")
    result2.set("MESSAGE IS:-\n" + lol2)
    return


bt2 = tkinter.Button(
    root,
    text="OK",
    bg="white",
    fg="black",
    bd=5,
    command=show2,
    font=("calibri", 15, "bold", "underline"),
)
bt2.place(x=10, y=320)
root.mainloop()
