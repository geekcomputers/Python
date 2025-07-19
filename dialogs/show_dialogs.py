import tkinter as tk
from tkinter import messagebox


def display_message_box() -> str | None:
    """Display a message box and return user's response"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Show a custom message box
    result = messagebox.askyesnocancel(
        title="Example Dialog Window", message="Do you want to continue?"
    )

    # Map boolean result to string values
    if result is None:
        return "cancel"
    elif result:
        return "yes"
    else:
        return "no"


if __name__ == "__main__":
    # Optional: Set font to ensure Chinese characters display correctly
    # import matplotlib.pyplot as plt
    # plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    # Display the message box
    response = display_message_box()
    print(f"User response: {response}")
