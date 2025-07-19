import tkinter as tk
from tkinter import Button, Label, OptionMenu, Scrollbar, StringVar, Text, Toplevel

from tkcalendar import Calendar  # Install via: pip install tkcalendar


def main() -> None:
    """Create and run the sports schedule management application"""
    # Create main application window
    window = tk.Tk()
    window.title("Sports Schedule Manager")
    window.geometry("600x500")
    window.resizable(True, True)  # Allow window resizing
    
    # Initialize list to store scheduled games
    game_list: list[str] = ["Game Schedule:"]
    
    # Available teams for dropdown selection
    team_options: list[str] = [
        "Eagles", "Tigers", "Bears", 
        "Sharks", "Falcons", "Dragons"
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
    # Configure theme colors
    primary_color = "#4A7A8C"
    secondary_color = "#F2E2D2"
    accent_color = "#D96941"
    text_color = "#262626"
    
    # Configure font styles
    title_font = ("Arial", 14, "bold")
    label_font = ("Arial", 10)
    button_font = ("Arial", 10, "bold")
    
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
    
    # Create left frame for team selection with styling
    left_frame = tk.Frame(window, padx=15, pady=15, bg=secondary_color)
    left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
    
    # Title label
    title_label = Label(
        left_frame, 
        text="Team Selection", 
        font=title_font, 
        bg=secondary_color, 
        fg=text_color
    )
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky="n")
    
    # Visitor team selection
    visitor_label = Label(
        left_frame, 
        text="Visitor Team:", 
        font=label_font, 
        bg=secondary_color, 
        fg=text_color
    )
    visitor_label.grid(row=1, column=0, sticky="w", pady=5)
    
    visitor_dropdown = OptionMenu(
        left_frame, 
        visitor_var, 
        *team_options
    )
    visitor_dropdown.config(
        font=label_font, 
        bg="white", 
        fg=text_color,
        relief=tk.RAISED,
        bd=2
    )
    visitor_dropdown["menu"].config(
        bg="white", 
        fg=text_color
    )
    visitor_dropdown.grid(row=1, column=1, sticky="ew", pady=5)
    
    # Home team selection
    home_label = Label(
        left_frame, 
        text="Home Team:", 
        font=label_font, 
        bg=secondary_color, 
        fg=text_color
    )
    home_label.grid(row=2, column=0, sticky="w", pady=5)
    
    home_dropdown = OptionMenu(
        left_frame, 
        home_var, 
        *team_options
    )
    home_dropdown.config(
        font=label_font, 
        bg="white", 
        fg=text_color,
        relief=tk.RAISED,
        bd=2
    )
    home_dropdown["menu"].config(
        bg="white", 
        fg=text_color
    )
    home_dropdown.grid(row=2, column=1, sticky="ew", pady=5)
    
    # Create calendar frame on the right with styling
    right_frame = tk.Frame(window, padx=15, pady=15, bg=secondary_color)
    right_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
    
    # Calendar title
    cal_title = Label(
        right_frame, 
        text="Select Date", 
        font=title_font, 
        bg=secondary_color, 
        fg=text_color
    )
    cal_title.pack(pady=(0, 10))
    
    calendar = Calendar(
        right_frame, 
        selectmode='day',
        year=2023,
        month=7,
        day=16,
        font=label_font,
        background=primary_color,
        foreground="white",
        selectbackground=accent_color,
        selectforeground="white",
        borderwidth=2,
        relief=tk.RAISED,
        locale='en_US'
    
    )
    calendar.pack(fill="both", expand=True)
    
    # Create game list display area with styling
    display_frame = tk.Frame(window, padx=15, pady=15, bg=secondary_color)
    display_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
    
    # Display title
    display_title = Label(
        display_frame, 
        text="Game Schedule", 
        font=title_font, 
        bg=secondary_color, 
        fg=text_color
    )
    display_title.pack(pady=(0, 10))
    
    # Text widget with scrollbar for game list
    game_display = Text(
        display_frame, 
        wrap=tk.WORD, 
        height=10,
        font=label_font,
        bg="white",
        fg=text_color,
        bd=2,
        relief=tk.SUNKEN
    )
    game_display.pack(side=tk.LEFT, fill="both", expand=True)
    
    scrollbar = Scrollbar(
        display_frame, 
        command=game_display.yview,
        bg=primary_color,
        troughcolor=secondary_color,
        relief=tk.FLAT
    )
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
        ),
        font=button_font,
        bg=accent_color,
        fg="white",
        activebackground="#BF4924",
        activeforeground="white",
        relief=tk.RAISED,
        bd=3
    )
    add_button.grid(row=3, column=0, columnspan=2, pady=20, sticky="ew")
    
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
    game_display: Text
) -> None:
    """Add a new game to the schedule and update the display"""
    # Get selected values
    visitor = visitor_var.get()
    home = home_var.get()
    date = calendar.get_date()
    
    # Validate input (prevent same team match)
    if visitor == home:
        show_error(window, "Input Error", "Visitor and home teams cannot be the same!")
        return
    
    # Create game entry and add to list
    game_entry = f"{visitor} vs {home} on {date}"
    game_list.append(game_entry)
    
    # Update the display with new list
    update_game_display(game_display, game_list)

def update_game_display(display: Text, game_list: list[str]) -> None:
    """Update the text widget with current game list"""
    # Clear existing content
    display.delete(1.0, tk.END)
    # Insert updated list
    display.insert(tk.END, "\n".join(game_list))

def show_error(window: tk.Tk, title: str, message: str) -> None:
    """Display an error message in a modal dialog"""
    error_window = Toplevel(window)
    error_window.title(title)
    error_window.geometry("350x180")
    error_window.resizable(False, False)
    error_window.configure(bg="#F8D7DA")  # Light red background
    
    # Center error window over main window
    error_window.geometry("+%d+%d" % (
        window.winfo_rootx() + window.winfo_width() // 2 - 175,
        window.winfo_rooty() + window.winfo_height() // 2 - 90
    ))
    
    # Error message frame
    message_frame = tk.Frame(error_window, bg="#F8D7DA")
    message_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Error message label
    message_label = Label(
        message_frame, 
        text=message, 
        font=("Arial", 10),
        bg="#F8D7DA",
        fg="#842029",
        wraplength=300
    )
    message_label.pack(fill="both", expand=True)
    
    # Close button
    close_button = Button(
        error_window, 
        text="OK", 
        command=error_window.destroy,
        font=("Arial", 10, "bold"),
        bg="#DC3545",
        fg="white",
        activebackground="#C82333",
        activeforeground="white",
        relief=tk.RAISED,
        bd=2,
        padx=15,
        pady=5
    )
    close_button.pack(pady=10)
    
    # Make dialog modal
    error_window.transient(window)
    error_window.grab_set()
    window.wait_window(error_window)

if __name__ == "__main__":
    main()    