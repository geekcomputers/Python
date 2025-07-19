import tkinter as tk
import tkinter.messagebox as mbox
from tkinter import Button, Entry, Event, Label, StringVar, Text
from typing import Any

import emoji


class Keypad(tk.Frame):
    """A custom keypad frame containing emoji buttons and control functions"""
    
    cells: list[list[str]] = [
        ["😀", "🥰", "😴", "🤓", "🤮", "🤬", "😨", "🤑", "😫", "😎"],
        [
            "🐒", "🐕", "🐎", "🐪", "🐁", "🐘", "🦘", "🦈", "🐓", "🐝",
            "👀", "🦴", "👩🏿", "‍🤝", "🧑", "🏾", "👱🏽", "‍♀", "🎞", "🎨", "⚽"
        ],
        [
            "🍕", "🍗", "🍜", "☕", "🍴", "🍉", "🍓", "🌴", "🌵", "🛺",
            "🚲", "🛴", "🚉", "🚀", "✈", "🛰", "🚦", "🏳", "‍🌈", "🌎", "🧭"
        ],
        [
            "🔥", "❄", "🌟", "🌞", "🌛", "🌝", "🌧", "🧺", "🧷", "🪒",
            "⛲", "🗼", "🕌", "👁", "‍🗨", "💬", "™", "💯", "🔕", "💥", "❤"
        ],
    ]

    def __init__(self, parent: tk.Tk, *args: Any, **kwargs: Any) -> None:
        """Initialize the keypad frame with emoji buttons and controls"""
        super().__init__(parent, *args, **kwargs)
        self.target: Entry | None = None
        self.memory: str = ""
        self.label: Label | None = None
        
        self._create_buttons()

    def _create_buttons(self) -> None:
        """Create and arrange all buttons in the keypad"""
        try:
            for row_idx, row in enumerate(self.cells):
                for col_idx, emoji_char in enumerate(row):
                    btn = Button(
                        self,
                        text=emoji_char,
                        command=lambda text=emoji_char: self.append(text),
                        font=("Arial", 14),
                        bg="yellow",
                        fg="blue",
                        borderwidth=3,
                        relief="raised"
                    )
                    btn.grid(row=row_idx, column=col_idx, sticky="news")

            control_buttons = [
                ("Space", self.space, 0, 10, 2),
                ("Tab", self.tab, 0, 12, 2),
                ("Backspace", self.backspace, 0, 14, 3),
                ("Clear", self.clear, 0, 17, 2),
                ("Hide", self.hide, 0, 19, 2)
            ]

            for text, cmd, row, col, colspan in control_buttons:
                btn = Button(
                    self,
                    text=text,
                    command=cmd,
                    font=("Arial", 14),
                    bg="yellow",
                    fg="blue",
                    borderwidth=3,
                    relief="raised"
                )
                btn.grid(row=row, column=col, columnspan=colspan, sticky="news")
                
        except Exception as e:
            print(f"Error creating keypad buttons: {str(e)}")
            # Optionally show error message to user
            # mbox.showerror("Error", f"Failed to initialize keypad: {str(e)}")

    def get(self) -> str | None:
        """Get current text from target entry widget"""
        try:
            if self.target:
                return self.target.get()
            return None
        except Exception as e:
            print(f"Error getting text: {str(e)}")
            return None

    def append(self, text: str) -> None:
        """Append text to target entry widget"""
        try:
            if self.target:
                self.target.insert("end", text)
        except Exception as e:
            print(f"Error appending text: {str(e)}")
            mbox.showerror("Error", f"Failed to append text: {str(e)}")

    def clear(self) -> None:
        """Clear all text from target entry widget"""
        try:
            if self.target:
                self.target.delete(0, tk.END)
        except Exception as e:
            print(f"Error clearing text: {str(e)}")
            mbox.showerror("Error", f"Failed to clear text: {str(e)}")

    def backspace(self) -> None:
        """Remove last character from target entry widget"""
        try:
            if self.target:
                current_text = self.get()
                if current_text:
                    new_text = current_text[:-1]
                    self.clear()
                    self.append(new_text)
        except Exception as e:
            print(f"Error during backspace: {str(e)}")
            mbox.showerror("Error", f"Failed to perform backspace: {str(e)}")

    def space(self) -> None:
        """Add a space to target entry widget"""
        try:
            if self.target:
                self.append(" ")
        except Exception as e:
            print(f"Error adding space: {str(e)}")
            mbox.showerror("Error", f"Failed to add space: {str(e)}")

    def tab(self) -> None:
        """Add 5 spaces (simulated tab) to target entry widget"""
        try:
            if self.target:
                self.append("     ")
        except Exception as e:
            print(f"Error adding tab: {str(e)}")
            mbox.showerror("Error", f"Failed to add tab: {str(e)}")

    def copy(self) -> None:
        """Copy current text to memory (clipboard simulation)"""
        try:
            if self.target:
                self.memory = self.get() or ""
                print(f"Copied to memory: {self.memory}")
        except Exception as e:
            print(f"Error copying text: {str(e)}")
            mbox.showerror("Error", f"Failed to copy text: {str(e)}")

    def paste(self) -> None:
        """Paste text from memory to target entry widget"""
        try:
            if self.target and self.memory:
                self.append(self.memory)
        except Exception as e:
            print(f"Error pasting text: {str(e)}")
            mbox.showerror("Error", f"Failed to paste text: {str(e)}")

    def show(self, entry: Entry) -> None:
        """Display keypad and set target entry widget"""
        try:
            self.target = entry
            self.place(relx=0.5, rely=0.6, anchor="c")
        except Exception as e:
            print(f"Error showing keypad: {str(e)}")
            mbox.showerror("Error", f"Failed to show keypad: {str(e)}")

    def hide(self) -> None:
        """Hide keypad and clear target entry widget"""
        try:
            self.target = None
            self.place_forget()
        except Exception as e:
            print(f"Error hiding keypad: {str(e)}")
            mbox.showerror("Error", f"Failed to hide keypad: {str(e)}")


def clear_text() -> None:
    """Clear both input entry and output text widgets"""
    try:
        inputentry.delete(0, tk.END)
        outputtxt.delete("1.0", tk.END)
    except Exception as e:
        print(f"Error clearing text: {str(e)}")
        mbox.showerror("Error", f"Failed to clear text fields: {str(e)}")


def search_emoji() -> None:
    """Search for meaning of entered emoji and display result"""
    try:
        emoji_input = inputentry.get()
        if not emoji_input:
            outputtxt.delete("1.0", tk.END)
            outputtxt.insert(tk.END, "No emoji entered. Please input an emoji first.")
            return
            
        # Check if input contains non-emoji characters
        if not all(emoji.is_emoji(char) for char in emoji_input):
            outputtxt.delete("1.0", tk.END)
            outputtxt.insert(tk.END, "Invalid input! Please enter only emojis.")
            return
            
        meaning = emoji.demojize(emoji_input)
        outputtxt.delete("1.0", tk.END)
        outputtxt.insert(tk.END, f"Meaning of Emoji: {emoji_input}\n\n{meaning}")
        
    except emoji.EmojiNotFoundError:
        outputtxt.delete("1.0", tk.END)
        outputtxt.insert(tk.END, "Emoji not recognized. Please try another emoji.")
    except Exception as e:
        print(f"Error processing emoji: {str(e)}")
        outputtxt.delete("1.0", tk.END)
        outputtxt.insert(tk.END, f"An error occurred: {str(e)}")


def exit_win() -> None:
    """Handle window closing confirmation"""
    try:
        if mbox.askokcancel("Exit", "Do you want to exit?"):
            window.destroy()
    except Exception as e:
        print(f"Error exiting application: {str(e)}")
        mbox.showerror("Error", f"Failed to exit application: {str(e)}")


def on_inputentry_click(event: Event) -> None:
    """Handle initial click on input entry to clear placeholder"""
    try:
        global firstclick1
        if firstclick1:
            firstclick1 = False
            inputentry.delete(0, tk.END)
    except Exception as e:
        print(f"Error handling input click: {str(e)}")


if __name__ == "__main__":
    try:
        # Initialize main window
        window: tk.Tk = tk.Tk()
        window.title("Emoji Dictionary")
        window.geometry("1000x700")

        # Title label
        title_label: Label = Label(
            window,
            text="EMOJI DICTIONARY",
            font=("Arial", 50, "underline"),
            fg="magenta"
        )
        title_label.place(x=160, y=10)

        # Instruction label
        input_label: Label = Label(
            window,
            text="Enter any Emoji you want to search...",
            font=("Arial", 30),
            fg="green"
        )
        input_label.place(x=160, y=120)

        # Input entry widget
        myname: StringVar = StringVar(window)
        firstclick1: bool = True
        inputentry: Entry = Entry(
            window,
            font=("Arial", 35),
            width=28,
            border=2,
            bg="light yellow",
            fg="brown",
            textvariable=myname
        )
        inputentry.insert(0, "Click to enter emoji...")
        inputentry.bind('<FocusIn>', on_inputentry_click)
        inputentry.place(x=120, y=180)

        # Search button
        search_btn: Button = Button(
            window,
            text="🔍 SEARCH",
            command=search_emoji,
            font=("Arial", 20),
            bg="light green",
            fg="blue",
            borderwidth=3,
            relief="raised"
        )
        search_btn.place(x=270, y=250)

        # Clear button
        clear_btn: Button = Button(
            window,
            text="🧹 CLEAR",
            command=clear_text,
            font=("Arial", 20),
            bg="orange",
            fg="blue",
            borderwidth=3,
            relief="raised"
        )
        clear_btn.place(x=545, y=250)

        # Output label
        output_label: Label = Label(
            window,
            text="Meaning...",
            font=("Arial", 30),
            fg="green"
        )
        output_label.place(x=160, y=340)

        # Output text widget
        outputtxt: Text = Text(
            window,
            height=7,
            width=57,
            font=("Arial", 17),
            bg="light yellow",
            fg="brown",
            borderwidth=3,
            relief="solid"
        )
        outputtxt.place(x=120, y=400)

        # Exit button
        exit_btn: Button = Button(
            window,
            text="❌ EXIT",
            command=exit_win,
            font=("Arial", 20),
            bg="red",
            fg="black",
            borderwidth=3,
            relief="raised"
        )
        exit_btn.place(x=435, y=610)

        # Initialize keypad
        keypad: Keypad = Keypad(window)

        # Keypad toggle button
        keypad_btn: Button = Button(
            window,
            text="⌨",
            command=lambda: keypad.show(inputentry),
            font=("Arial", 18),
            bg="light yellow",
            fg="green",
            borderwidth=3,
            relief="raised"
        )
        keypad_btn.place(x=870, y=183)

        # Configure window close protocol
        window.protocol("WM_DELETE_WINDOW", exit_win)
        
        # Start main event loop
        window.mainloop()
        
    except Exception as e:
        print(f"Fatal error during application initialization: {str(e)}")
        # Consider showing a critical error message here
        # mbox.showerror("Critical Error", f"Failed to start application: {str(e)}")