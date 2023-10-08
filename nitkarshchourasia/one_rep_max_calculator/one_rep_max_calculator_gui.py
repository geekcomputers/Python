import tkinter as tk


class OneRepMaxCalculator:
    """
    A class used to calculate the estimated one-repetition maximum (1RM) for a weightlifting exercise.

    Attributes
    ----------
    window : tk.Tk
        The main window of the application.
    weight_entry : tk.Entry
        Entry field to input the weight lifted.
    rep_entry : tk.Entry
        Entry field to input the number of reps performed.
    result_value_label : tk.Label
        Label to display the calculated 1RM.

    Methods
    -------
    calculate_1rm():
        Calculates the estimated 1RM based on the Epley formula.
    display_result():
        Displays the calculated 1RM in the application window.
    run():
        Runs the application.
    """

    def __init__(self):
        """Initializes the OneRepMaxCalculator with a window and widgets."""
        self.window = tk.Tk()
        self.window.title("One-Rep Max Calculator")
        self.window.geometry("300x150")

        # Create and pack widgets
        tk.Label(self.window, text="Enter the weight you lifted (in kg):").pack()
        self.weight_entry = tk.Entry(self.window)
        self.weight_entry.pack()

        tk.Label(self.window, text="Enter the number of reps you performed:").pack()
        self.rep_entry = tk.Entry(self.window)
        self.rep_entry.pack()

        tk.Button(self.window, text="Calculate", command=self.display_result).pack()

        tk.Label(self.window, text="Your estimated one-rep max (1RM):").pack()
        self.result_value_label = tk.Label(self.window)
        self.result_value_label.pack()

    def calculate_1rm(self):
        """Calculates and returns the estimated 1RM."""
        weight = int(self.weight_entry.get())
        reps = int(self.rep_entry.get())
        return (weight * reps * 0.0333) + weight

    def display_result(self):
        """Calculates the 1RM and updates result_value_label with it."""
        one_rep_max = self.calculate_1rm()
        self.result_value_label.config(text=f"{one_rep_max} kg")

    def run(self):
        """Runs the Tkinter event loop."""
        self.window.mainloop()


# Usage
if __name__ == "__main__":
    calculator = OneRepMaxCalculator()
    calculator.run()

# Improve the program.
# Make the fonts, bigger.
# - Use text formatting...
# Use dark mode.
# Have an option to use dark mode and light mode.
