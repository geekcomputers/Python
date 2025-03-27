from tkinter import *
import pandas as pd
import random
import os

# Constants
FRONT_COLOR = "#B1DDC6"  # Green background for front
BACK_COLOR = "#F08080"   # Red background for back
BUTTON_RED = "#FF5555"   # Bright red for buttons
BUTTON_GREEN = "#4CAF50"  # Green for tick and wrong buttons
FLIP_TIME_MS = 3000  # Time before card flips (3 seconds)
FONT_NAME = "Arial"
WINDOW_WIDTH = 400  # Further reduced window width
WINDOW_HEIGHT = 300  # Further reduced window height
PADDING = 20  # Reduced padding

# ---------------------------- DATA SETUP ------------------------------- #
def load_data():
    """Load word data from CSV file"""
    try:
        # Try to load from word_to_learn.csv first
        data_file = "data/word_to_learn.csv"
        data = pd.read_csv(data_file)
        
        # If file exists but is empty or has only headers
        if len(data) == 0:
            raise pd.errors.EmptyDataError("Empty file")
            
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # If word_to_learn.csv doesn't exist or is empty, load from translation.txt
        source_file = "data/translation.txt"
        source_data = pd.read_csv(source_file)
        
        # Save all data to word_to_learn.csv
        data_file = "data/word_to_learn.csv"
        source_data.to_csv(data_file, index=False)
        data = source_data
    
    return data.to_dict(orient='records'), data_file

to_learn, data_file = load_data()
current_card = {}
flip_timer = None

# ---------------------------- CARD FUNCTIONS ------------------------------- #
def next_card():
    """Display a new random English word from the word list"""
    global current_card, flip_timer
    
    # Cancel previous timer if exists
    if flip_timer:
        window.after_cancel(flip_timer)
        
    # Check if we have words left to learn
    if not to_learn:
        show_completion_message()
        return
        
    # Select a random word and display on card front
    current_card = random.choice(to_learn)
    canvas.config(bg=FRONT_COLOR)
    canvas.itemconfig(card_title, text="English", fill="black")
    canvas.itemconfig(card_word, text=current_card["English"], fill="black")
    
    # Set timer to flip card after delay
    flip_timer = window.after(FLIP_TIME_MS, flip_card)

def flip_card():
    """Flip the card to show the Hindi translation"""
    canvas.config(bg=BACK_COLOR)
    canvas.itemconfig(card_title, text="Hindi", fill="white")
    canvas.itemconfig(card_word, text=current_card["Hindi"], fill="white")

def remove_card():
    """Remove the current card from the deck when user knows it (✓ button)"""
    global to_learn
    
    # Remove current card from the deck
    to_learn.remove(current_card)
    
    # Save updated word list to word_to_learn.csv
    pd.DataFrame(to_learn).to_csv(data_file, index=False)
    
    # Show next card
    next_card()

def skip_card():
    """Skip current card (❌ button) without removing it from the deck"""
    # Just move to the next card without changing the deck
    next_card()

def show_completion_message():
    """Display completion message when all words are learned"""
    canvas.config(bg=FRONT_COLOR)
    canvas.itemconfig(card_title, text="Congratulations!", fill="black")
    canvas.itemconfig(card_word, text="You've learned all the words!", fill="black")
    
    # Disable buttons when complete
    wrong_button.config(state="disabled")
    right_button.config(state="disabled")

def reset_progress():
    """Reset progress by reloading all words from translation.txt"""
    global to_learn, data_file
    
    # Reload original word list from translation.txt
    source_file = "data/translation.txt"
    source_data = pd.read_csv(source_file)
    
    # Save all data to word_to_learn.csv
    data_file = "data/word_to_learn.csv"
    source_data.to_csv(data_file, index=False)
    
    # Update current deck
    to_learn = source_data.to_dict(orient='records')
    
    # Enable buttons
    wrong_button.config(state="normal")
    right_button.config(state="normal")
    
    # Show a new card
    next_card()

# ------------------------------- UI SETUP ------------------------------- #
window = Tk()
window.title("Flashy - English-Hindi Learning App")
window.config(background=FRONT_COLOR, padx=PADDING, pady=PADDING)

# Canvas for the flashcard - no card images, just colored background
canvas = Canvas(width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg=FRONT_COLOR, highlightthickness=1, highlightbackground="gray")

# Text elements on the card
card_title = canvas.create_text(WINDOW_WIDTH//2, WINDOW_HEIGHT//3, text="", font=(FONT_NAME, 24, "italic"))
card_word = canvas.create_text(WINDOW_WIDTH//2, WINDOW_HEIGHT//2, text="", font=(FONT_NAME, 36, "bold"))
canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Buttons
wrong_button = Button(
    text="❌", 
    font=(FONT_NAME, 16),
    highlightthickness=0,
    bg=BUTTON_GREEN,
    fg="white",
    activebackground="#3E8E41",  # Darker green when clicked
    activeforeground="white",
    border=0,
    command=skip_card
)
wrong_button.grid(row=1, column=0, pady=10)

right_button = Button(
    text="✓", 
    font=(FONT_NAME, 16),
    highlightthickness=0,
    bg=BUTTON_GREEN,
    fg="white",
    activebackground="#3E8E41",  # Darker green when clicked
    activeforeground="white",
    border=0,
    command=remove_card
)
right_button.grid(column=1, row=1, pady=10)

# Bottom row buttons
button_frame = Frame(window, bg=FRONT_COLOR)
button_frame.grid(row=2, column=0, columnspan=2, pady=10)

# Reset button
reset_button = Button(
    button_frame,
    text="Reset Progress",
    highlightthickness=0,
    bg=BUTTON_RED,
    fg="white",
    activebackground="#FF3333",
    activeforeground="white",
    command=reset_progress
)
reset_button.grid(row=0, column=0, padx=5)

# Exit button
exit_button = Button(
    button_frame,
    text="Exit",
    highlightthickness=0,
    bg=BUTTON_RED,
    fg="white",
    activebackground="#FF3333",
    activeforeground="white",
    command=window.destroy
)
exit_button.grid(row=0, column=1, padx=5)

# Initialize with the first card
next_card()

window.mainloop()

