from tkinter import messagebox
import tkinter as tk


# Function to be called when button is clicked
def add_Button():
    task = Input.get()
    if task:
        List.insert(tk.END, task)
        Input.delete(0, tk.END)


def del_Button():
    try:
        task = List.curselection()[0]
        List.delete(task)
    except IndexError:
        messagebox.showwarning("Selection Error", "Please select a task to delete.")


# Create the main window
window = tk.Tk()
window.title("Task Manager")
window.geometry("500x500")
window.resizable(False, False)
window.config(bg="light grey")

# text filed
Input = tk.Entry(window, width=50)
Input.grid(row=0, column=0, padx=20, pady=60)
Input.focus()

# Create the button
add = tk.Button(window, text="ADD TASK", height=2, width=9, command=add_Button)
add.grid(row=0, column=1, padx=20, pady=0)

delete = tk.Button(window, text="DELETE TASK", height=2, width=10, command=del_Button)
delete.grid(row=1, column=1)

# creating list box
List = tk.Listbox(window, width=50, height=20)
List.grid(row=1, column=0)


window.mainloop()
