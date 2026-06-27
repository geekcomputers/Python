import tkinter as tk
from tkinter import messagebox
from tkcalendar import Calendar

def show():
    visitor = selected1.get()
    home = selected2.get()
    
    # Validation: Check if teams are the same
    if visitor == home:
        messagebox.showwarning("Input Error", "Visitor and Home teams cannot be the same!")
        return

    game = f"{visitor} vs {home} on {cal.get_date()}"
    
    # Update the Text widget
    display_area.config(state=tk.NORMAL) # Enable editing
    display_area.insert(tk.END, game + "\n")
    display_area.config(state=tk.DISABLED) # Make read-only
    
    # Optional: Clear dropdowns or reset
    selected1.set("Team 1")
    selected2.set("Team 2")

window = tk.Tk()
window.title("Game Scheduler")
window.geometry("600x550")

options = ["Team 1", "Team 2", "Team 3", "Team 4", "Team 5", "Team 6"]
selected1 = tk.StringVar(value="Team 1")
selected2 = tk.StringVar(value="Team 2")

# UI Layout using Grid for better alignment
tk.Label(window, text="Visitor:").grid(row=0, column=0, padx=10, pady=10)
tk.OptionMenu(window, selected1, *options).grid(row=0, column=1)

tk.Label(window, text="Home:").grid(row=1, column=0, padx=10, pady=10)
tk.OptionMenu(window, selected2, *options).grid(row=1, column=1)

cal = Calendar(window, selectmode="day", year=2026, month=6, day=2)
cal.grid(row=0, column=2, rowspan=2, padx=20)

tk.Button(window, text="Add to Schedule", command=show).grid(row=2, column=0, columnspan=2, pady=20)

# Scrollable display area
display_area = tk.Text(window, height=10, width=50, state=tk.DISABLED)
display_area.grid(row=3, column=0, columnspan=3, padx=10)

window.mainloop()
