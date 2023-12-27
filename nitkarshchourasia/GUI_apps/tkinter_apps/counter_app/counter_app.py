import tkinter as tk
from tkinter import ttk

# Creating a counter app using tkinter


class MyApplication:
    def __init__(self, master):
        self.master = master
        self.master.title("Counter App")
        self.master.geometry("300x300")

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.master)
        frame.pack(padx=20, pady=20)

        self.label = ttk.Label(frame, text="0", font=("Arial Bold", 70))
        self.label.grid(row=0, column=0, padx=20, pady=20)

        add_button = ttk.Button(frame, text="Add", command=self.on_add_click)
        add_button.grid(row=1, column=0, pady=10)

        remove_button = ttk.Button(frame, text="Remove", command=self.on_remove_click)
        remove_button.grid(row=2, column=0, pady=10)

    def on_add_click(self):
        current_text = self.label.cget("text")
        new_text = int(current_text) + 1
        self.label.config(text=new_text)

    def on_remove_click(self):
        current_text = self.label.cget("text")
        new_text = int(current_text) - 1
        self.label.config(text=new_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = MyApplication(root)
    root.mainloop()
