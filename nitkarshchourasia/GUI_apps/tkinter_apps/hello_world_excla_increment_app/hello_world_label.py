import tkinter as tk
from tkinter import ttk


class MyApplication:
    def __init__(self, master):
        self.master = master
        # Want to understand why .master.title was used?
        self.master.title("Hello World")

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.master)
        frame.pack(padx=20, pady=20)
        # grid and pack are different geometry managers.
        self.label = ttk.Label(frame, text="Hello World!", font=("Arial Bold", 50))
        self.label.grid(row=0, column=0, padx=20, pady=20)

        # Add a button for interaction
        concat_button = ttk.Button(
            frame, text="Click Me!", command=self.on_button_click
        )
        concat_button.grid(row=1, column=0, pady=10)

        remove_button = ttk.Button(
            frame, text="Remove '!'", command=self.on_remove_click
        )
        remove_button.grid(row=2, column=0, pady=10)

    def on_button_click(self):
        current_text = self.label.cget("text")
        # current_text = self.label["text"]
        #! Solve this.
        new_text = current_text + "!"
        self.label.config(text=new_text)

    def on_remove_click(self):
        # current_text = self.label.cget("text")
        current_text = self.label["text"]
        #! Solve this.
        new_text = current_text[:-1]
        self.label.config(text=new_text)
        # TODO: Can make a char matching function, to remove the last char, if it is a '!'.


if __name__ == "__main__":
    root = tk.Tk()
    app = MyApplication(root)
    root.mainloop()
