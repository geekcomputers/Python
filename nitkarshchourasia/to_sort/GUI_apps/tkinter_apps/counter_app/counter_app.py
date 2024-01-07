# Author: Nitkarsh Chourasia
# Date created: 28/12/2023

# Import the required libraries
import tkinter as tk
from tkinter import ttk


class MyApplication:
    """A class to create a counter app."""

    def __init__(self, master):
        # Initialize the master window
        self.master = master
        # Set the title and geometry of the master window
        self.master.title("Counter App")
        self.master.geometry("300x300")

        # Create the widgets
        self.create_widgets()

    # Create the widgets
    def create_widgets(self):
        # Create a frame to hold the widgets
        frame = ttk.Frame(self.master)
        # Pack the frame to the master window
        frame.pack(padx=20, pady=20)

        # Create a label to display the counter
        self.label = ttk.Label(frame, text="0", font=("Arial Bold", 70))
        # Grid the label to the frame
        self.label.grid(row=0, column=0, padx=20, pady=20)

        # Add a button for interaction to increase the counter
        add_button = ttk.Button(frame, text="Add", command=self.on_add_click)
        # Grid the button to the frame
        add_button.grid(row=1, column=0, pady=10)

        # Add a button for interaction to decrease the counter
        remove_button = ttk.Button(frame, text="Remove", command=self.on_remove_click)
        # Grid the button to the frame
        remove_button.grid(row=2, column=0, pady=10)

    # Add a click event handler
    def on_add_click(self):
        # Get the current text of the label
        current_text = self.label.cget("text")
        # Convert the text to an integer and add 1
        new_text = int(current_text) + 1
        # Set the new text to the label
        self.label.config(text=new_text)

    # Add a click event handler
    def on_remove_click(self):
        # Get the current text of the label
        current_text = self.label.cget("text")
        # Convert the text to an integer and subtract 1
        new_text = int(current_text) - 1
        # Set the new text to the label
        self.label.config(text=new_text)


if __name__ == "__main__":
    # Create the root window
    root = tk.Tk()
    # Create an instance of the application
    app = MyApplication(root)
    # Run the app
    root.mainloop()
