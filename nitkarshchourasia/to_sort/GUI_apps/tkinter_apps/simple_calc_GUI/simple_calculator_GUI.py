from tkinter import *

# To install hupper, use: "pip install hupper"
# On CMD, or Terminal.
import hupper


# Python program to create a simple GUI
# calculator using Tkinter

# Importing everything from tkinter module

# globally declare the expression variable
# Global variables are those variables that can be accessed and used inside any function.
global expression, equation
expression = ""


def start_reloader():
    """Adding a live server for tkinter test GUI, which reloads the GUI when the code is changed."""
    reloader = hupper.start_reloader("p1.main")


# Function to update expression
# In the text entry box
def press(num):
    """Function to update expression in the text entry box.

    Args:
        num (int): The number to be input to the expression.
    """
    # point out the global expression variable
    global expression, equation

    # concatenation of string
    expression = expression + str(num)

    # update the expression by using set method
    equation.set(expression)


# Function to evaluate the final expression
def equalpress():
    """Function to evaluate the final expression."""
    # Try and except statement is used
    # For handling the errors like zero
    # division error etc.

    # Put that code inside the try block
    # which may generate the error

    try:
        global expression, equation
        # eval function evaluate the expression
        # and str function convert the result
        # into string

        #! Is using eval() function, safe?
        #! Isn't it a security risk?!

        total = str(eval(expression))
        equation.set(total)

        # Initialize the expression variable
        # by empty string

        expression = ""

    # if error is generate then handle
    # by the except block

    except:
        equation.set(" Error ")
        expression = ""


# Function to clear the contents
# of text entry box


def clear_func():
    """Function to clear the contents of text entry box."""
    global expression, equation
    expression = ""
    equation.set("")


def close_app():
    """Function to close the app."""
    global gui  # Creating a global variable
    return gui.destroy()


# Driver code
def main():
    """Driver code for the GUI calculator."""
    # create a GUI window

    global gui  # Creating a global variable
    gui = Tk()
    global equation
    equation = StringVar()

    # set the background colour of GUI window
    gui.configure(background="grey")

    # set the title of GUI window
    gui.title("Simple Calculator")

    # set the configuration of GUI window
    gui.geometry("270x160")

    # StringVar() is the variable class
    # we create an instance of this class

    # create the text entry box for
    # showing the expression .

    expression_field = Entry(gui, textvariable=equation)

    # grid method is used for placing
    # the widgets at respective positions
    # In table like structure.

    expression_field.grid(columnspan=4, ipadx=70)

    # create a Buttons and place at a particular
    # location inside the root windows.
    # when user press the button, the command or
    # function affiliated to that button is executed.

    # Embedding buttons to the GUI window.
    # Button 1 = int(1)
    button1 = Button(
        gui,
        text=" 1 ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press(1),
        height=1,
        width=7,
    )
    button1.grid(row=2, column=0)

    # Button 2 = int(2)
    button2 = Button(
        gui,
        text=" 2 ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press(2),
        height=1,
        width=7,
    )
    button2.grid(row=2, column=1)

    # Button 3 = int(3)
    button3 = Button(
        gui,
        text=" 3 ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press(3),
        height=1,
        width=7,
    )
    button3.grid(row=2, column=2)

    # Button 4 = int(4)
    button4 = Button(
        text=" 4 ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press(4),
        height=1,
        width=7,
    )
    button4.grid(row=3, column=0)

    # Button 5 = int(5)
    button5 = Button(
        text=" 5 ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press(5),
        height=1,
        width=7,
    )
    button5.grid(row=3, column=1)

    # Button 6 = int(6)
    button6 = Button(
        text=" 6 ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press(6),
        height=1,
        width=7,
    )
    button6.grid(row=3, column=2)

    # Button 7 = int(7)
    button7 = Button(
        text=" 7 ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press(7),
        height=1,
        width=7,
    )
    button7.grid(row=4, column=0)

    # Button 8 = int(8)
    button8 = Button(
        text=" 8 ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press(8),
        height=1,
        width=7,
    )
    button8.grid(row=4, column=1)

    # Button 9 = int(9)
    button9 = Button(
        text=" 9 ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press(9),
        height=1,
        width=7,
    )
    button9.grid(row=4, column=2)

    # Button 0 = int(0)
    button0 = Button(
        text=" 0 ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press(0),
        height=1,
        width=7,
    )
    button0.grid(row=5, column=0)

    # Embedding the operator buttons.

    # Button + = inputs "+" operator.
    plus = Button(
        gui,
        text=" + ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press("+"),
        height=1,
        width=7,
    )
    plus.grid(row=2, column=3)

    # Button - = inputs "-" operator.
    minus = Button(
        gui,
        text=" - ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press("-"),
        height=1,
        width=7,
    )
    minus.grid(row=3, column=3)

    # Button * = inputs "*" operator.
    multiply = Button(
        gui,
        text=" * ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press("*"),
        height=1,
        width=7,
    )
    multiply.grid(row=4, column=3)

    # Button / = inputs "/" operator.
    divide = Button(
        gui,
        text=" / ",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press("/"),
        height=1,
        width=7,
    )
    divide.grid(row=5, column=3)

    # Button = = inputs "=" operator.
    equal = Button(
        gui,
        text=" = ",
        fg="#FFFFFF",
        bg="#000000",
        command=equalpress,
        height=1,
        width=7,
    )
    equal.grid(row=5, column=2)

    # Button Clear = clears the input field.
    clear = Button(
        gui,
        text="Clear",
        fg="#FFFFFF",
        bg="#000000",
        command=clear_func,
        height=1,
        width=7,
    )
    clear.grid(row=5, column=1)  # Why this is an in string, the column?

    # Button . = inputs "." decimal in calculations.
    Decimal = Button(
        gui,
        text=".",
        fg="#FFFFFF",
        bg="#000000",
        command=lambda: press("."),
        height=1,
        width=7,
    )
    Decimal.grid(row=6, column=0)

    # gui.after(1000, lambda: gui.focus_force()) # What is this for?
    # gui.after(1000, close_app)

    gui.mainloop()


class Metadata:
    def __init__(self):
        # Author Information
        self.author_name = "Nitkarsh Chourasia"
        self.author_email = "playnitkarsh@gmail.com"
        self.gh_profile_url = "https://github.com/NitkarshChourasia"
        self.gh_username = "NitkarshChourasia"

        # Project Information
        self.project_name = "Simple Calculator"
        self.project_description = (
            "A simple calculator app made using Python and Tkinter."
        )
        self.project_creation_date = "30-09-2023"
        self.project_version = "1.0.0"

        # Edits
        self.original_author = "Nitkarsh Chourasia"
        self.original_author_email = "playnitkarsh@gmail.com"
        self.last_edit_date = "30-09-2023"
        self.last_edit_author = "Nitkarsh Chourasia"
        self.last_edit_author_email = "playnitkarsh@gmail.com"
        self.last_edit_author_gh_profile_url = "https://github.com/NitkarshChourasia"
        self.last_edit_author_gh_username = "NitkarshChourasia"

    def display_author_info(self):
        """Display author information."""
        print(f"Author Name: {self.author_name}")
        print(f"Author Email: {self.author_email}")
        print(f"GitHub Profile URL: {self.gh_profile_url}")
        print(f"GitHub Username: {self.gh_username}")

    def display_project_info(self):
        """Display project information."""
        print(f"Project Name: {self.project_name}")
        print(f"Project Description: {self.project_description}")
        print(f"Project Creation Date: {self.project_creation_date}")
        print(f"Project Version: {self.project_version}")

    def display_edit_info(self):
        """Display edit information."""
        print(f"Original Author: {self.original_author}")
        print(f"Original Author Email: {self.original_author_email}")
        print(f"Last Edit Date: {self.last_edit_date}")
        print(f"Last Edit Author: {self.last_edit_author}")
        print(f"Last Edit Author Email: {self.last_edit_author_email}")
        print(
            f"Last Edit Author GitHub Profile URL: {self.last_edit_author_gh_profile_url}"
        )
        print(f"Last Edit Author GitHub Username: {self.last_edit_author_gh_username}")

    def open_github_profile(self) -> None:
        """Open the author's GitHub profile in a new tab."""
        import webbrowser

        return webbrowser.open_new_tab(self.gh_profile_url)


if __name__ == "__main__":
    # start_reloader()
    main()

    # # Example usage:
    # metadata = Metadata()

    # # Display author information
    # metadata.display_author_info()

    # # Display project information
    # metadata.display_project_info()

    # # Display edit information
    # metadata.display_edit_info()

# TODO: More features to add:
# Responsive design is not there.
# The program is not OOP based, there is lots and lots of repetitions.
# Bigger fonts.
# Adjustable everything.
# Default size, launch, but customizable.
# Adding history.
# Being able to continuosly operate on a number.
# What is the error here, see to it.
# To add Author Metadata.

# TODO: More features will be added, soon.


# Working.
# Perfect.
# Complete.
# Do not remove the comments, they make the program understandable.
# Thank you. :) ❤️
# Made with ❤️
