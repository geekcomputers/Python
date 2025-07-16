import tkinter as tk
from tkinter import StringVar, Label, Button, OptionMenu
from tkcalendar import Calendar  # Install via: pip install tkcalendar

def main() -> None:
    """Create and run the sports schedule management application"""
    # Create main application window
    window = tk.Tk()
    window.title("Sports Schedule Manager")
    window.geometry("600x500")
    window.resizable(True, True)  # Allow window resizing
    
    # Initialize list to store scheduled games
    game_list: list[str] = ["Game List:"]
    
    # Available teams for dropdown selection
    team_options: list[str] = [
        "Team 1", "Team 2", "Team 3", 
        "Team 4", "Team 5", "Team 6"
    ]
    
    # Create and arrange all GUI components
    create_widgets(window, game_list, team_options)
    
    # Start the main event loop
    window.mainloop()

def create_widgets(
    window: tk.Tk, 
    game_list: list[str], 
    team_options: list[str]
) -> None:
    """Create and position all GUI widgets in the main window"""
    # Variables to store selected teams
    visitor_var: StringVar = StringVar(window)
    home_var: StringVar = StringVar(window)
    
    # Set default selections
    visitor_var.set(team_options[0])
    home_var.set(team_options[1])
    
    # Configure grid weights for responsive layout
    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=1)
    window.rowconfigure(0, weight=1)
    window.rowconfigure(1, weight=1)
    window.rowconfigure(2, weight=3)
    
    # Create left frame for team selection
    left_frame = tk.Frame(window, padx=10, pady=10)
    left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
    
    # Visitor team selection
    visitor_label = Label(left_frame, text="Visitor:")
    visitor_label.grid(row=0, column=0, sticky="w", pady=5)
    
    visitor_dropdown = OptionMenu(left_frame, visitor_var, *team_options)
    visitor_dropdown.grid(row=0, column=1, sticky="ew", pady=5)
    
    # Home team selection
    home_label = Label(left_frame, text="Home:")
    home_label.grid(row=1, column=0, sticky="w", pady=5)
    
    home_dropdown = OptionMenu(left_frame, home_var, *team_options)
    home_dropdown.grid(row=1, column=1, sticky="ew", pady=5)
    
    # Create calendar frame on the right
    right_frame = tk.Frame(window, padx=10, pady=10)
    right_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
    
    calendar = Calendar(
        right_frame, 
        selectmode='day',
        year=2023,
        month=7,
        day=16
    )
    calendar.pack(fill="both", expand=True)
    
    # Create game list display area
    display_frame = tk.Frame(window, padx=10, pady=10)
    display_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
    
    # Text widget with scrollbar for game list
    game_display = tk.Text(display_frame, wrap=tk.WORD, height=10)
    game_display.pack(side=tk.LEFT, fill="both", expand=True)
    
    scrollbar = tk.Scrollbar(display_frame, command=game_display.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    game_display.config(yscrollcommand=scrollbar.set)
    
    # Initialize display with empty list
    update_game_display(game_display, game_list)
    
    # Add to schedule button - pass game_display to add_game
    add_button = Button(
        left_frame, 
        text="Add to Schedule", 
        command=lambda: add_game(
            window, game_list, visitor_var, home_var, calendar, game_display
        )
    )
    add_button.grid(row=2, column=0, columnspan=2, pady=20)
    
    # Configure weights for responsive resizing
    left_frame.columnconfigure(1, weight=1)
    right_frame.columnconfigure(0, weight=1)
    right_frame.rowconfigure(0, weight=1)
    display_frame.columnconfigure(0, weight=1)
    display_frame.rowconfigure(0, weight=1)

def add_game(
    window: tk.Tk,
    game_list: list[str],
    visitor_var: StringVar,
    home_var: StringVar,
    calendar: Calendar,
    game_display: tk.Text  # Added game_display parameter
) -> None:
    """Add a new game to the schedule and update the display"""
    # Get selected values
    visitor = visitor_var.get()
    home = home_var.get()
    date = calendar.get_date()
    
    # Validate input (prevent same team match)
    if visitor == home:
        show_error(window, "Error", "Visitor and home teams cannot be the same!")
        return
    
    # Create game entry and add to list
    game_entry = f"{visitor} vs {home} on {date}"
    game_list.append(game_entry)
    
    # Update the display with new list
    update_game_display(game_display, game_list)

def update_game_display(display: tk.Text, game_list: list[str]) -> None:
    """Update the text widget with current game list"""
    # Clear existing content
    display.delete(1.0, tk.END)
    # Insert updated list
    display.insert(tk.END, "\n".join(game_list))

def show_error(window: tk.Tk, title: str, message: str) -> None:
    """Display an error message in a modal dialog"""
    error_window = tk.Toplevel(window)
    error_window.title(title)
    error_window.geometry("300x150")
    error_window.resizable(False, False)
    
    # Center error window over main window
    error_window.geometry("+%d+%d" % (
        window.winfo_rootx() + window.winfo_width() // 2 - 150,
        window.winfo_rooty() + window.winfo_height() // 2 - 75
    ))
    
    # Error message label
    message_label = Label(error_window, text=message, padx=20, pady=20)
    message_label.pack(fill="both", expand=True)
    
    # Close button
    close_button = Button(error_window, text="OK", command=error_window.destroy)
    close_button.pack(pady=10)
    
    # Make dialog modal
    error_window.transient(window)
    error_window.grab_set()
    window.wait_window(error_window)

if __name__ == "__main__":
    main()