import tkinter as tk
from tkinter import messagebox, ttk


class OneRepMaxCalculator:
    """
    A GUI application to calculate the estimated one-repetition maximum (1RM)
    using multiple formulas with dark/light mode support.
    """

    def __init__(self) -> None:
        """Initialize the calculator with dark mode as default."""
        self.window = tk.Tk()
        self.window.title("One-Rep Max Calculator")
        self.window.geometry("450x300")
        self.is_dark_mode = True

        # Configure fonts
        self.default_font = ("Arial", 12)
        self.title_font = ("Arial", 16, "bold")
        self.button_font = ("Arial", 12, "bold")

        # Available 1RM formulas
        self.formulas = {
            "Epley": lambda w, r: w * (1 + r * 0.0333),
            "Brzycki": lambda w, r: w / (1.0278 - 0.0278 * r),
            "Lombardi": lambda w, r: w * (r**0.1),
            "Mayhew": lambda w, r: (100 * w)
            / (52.2 + 41.9 * (2.71828 ** (-0.055 * r))),
        }

        # Apply dark mode initially
        self.apply_dark_mode()

        # Create widgets
        self.create_widgets()

    def create_widgets(self) -> None:
        """Create and layout GUI widgets with validation checks."""
        # Title
        title_label = tk.Label(
            self.window,
            text="One-Rep Max Calculator",
            font=self.title_font,
            bg=self.bg_color,
            fg=self.text_color,
        )
        title_label.pack(pady=10)

        # Weight input with validation
        weight_frame = tk.Frame(self.window, bg=self.bg_color)
        weight_frame.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(
            weight_frame,
            text="Weight Lifted (kg):",
            font=self.default_font,
            bg=self.bg_color,
            fg=self.text_color,
            width=18,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        self.weight_entry = tk.Entry(
            weight_frame,
            font=self.default_font,
            bg=self.entry_bg,
            fg=self.text_color,
            insertbackground=self.text_color,
            validate="key",
            validatecommand=(self.window.register(self.validate_numeric_input), "%P"),
        )
        self.weight_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Reps input with validation
        reps_frame = tk.Frame(self.window, bg=self.bg_color)
        reps_frame.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(
            reps_frame,
            text="Number of Reps:",
            font=self.default_font,
            bg=self.bg_color,
            fg=self.text_color,
            width=18,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        self.rep_entry = tk.Entry(
            reps_frame,
            font=self.default_font,
            bg=self.entry_bg,
            fg=self.text_color,
            insertbackground=self.text_color,
            validate="key",
            validatecommand=(self.window.register(self.validate_integer_input), "%P"),
        )
        self.rep_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Formula selection
        formula_frame = tk.Frame(self.window, bg=self.bg_color)
        formula_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            formula_frame,
            text="Formula:",
            font=self.default_font,
            bg=self.bg_color,
            fg=self.text_color,
            width=18,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        self.formula_var = tk.StringVar(value="Epley")
        formula_menu = ttk.Combobox(
            formula_frame,
            textvariable=self.formula_var,
            values=list(self.formulas.keys()),
            font=self.default_font,
            state="readonly",
            width=15,
        )
        formula_menu.pack(side=tk.LEFT)

        # Button frame
        button_frame = tk.Frame(self.window, bg=self.bg_color)
        button_frame.pack(pady=15)

        # Calculate button
        calculate_btn = tk.Button(
            button_frame,
            text="Calculate 1RM",
            command=self.display_result,
            font=self.button_font,
            bg=self.button_bg,
            fg=self.button_fg,
            activebackground=self.button_active_bg,
            activeforeground=self.button_fg,
            padx=15,
        )
        calculate_btn.pack(side=tk.LEFT, padx=10)

        # Theme toggle button
        theme_btn = tk.Button(
            button_frame,
            text="Toggle Theme",
            command=self.toggle_theme,
            font=self.button_font,
            bg=self.button_bg,
            fg=self.button_fg,
            activebackground=self.button_active_bg,
            activeforeground=self.button_fg,
            padx=15,
        )
        theme_btn.pack(side=tk.LEFT, padx=10)

        # Result display
        result_frame = tk.Frame(self.window, bg=self.bg_color)
        result_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            result_frame,
            text="Estimated 1RM:",
            font=self.default_font,
            bg=self.bg_color,
            fg=self.text_color,
        ).pack(side=tk.LEFT, padx=5)

        self.result_value_label = tk.Label(
            result_frame,
            text="",
            font=self.title_font,
            bg=self.bg_color,
            fg="#22A7F0",  # Accent color for result
        )
        self.result_value_label.pack(side=tk.LEFT, padx=10)

        # Formula info
        self.formula_info_label = tk.Label(
            self.window,
            text="Formula: Epley (w × (1 + r × 0.0333))",
            font=("Arial", 10),
            bg=self.bg_color,
            fg="#888888",
        )
        self.formula_info_label.pack(pady=5)

        # Bind formula change event
        formula_menu.bind("<<ComboboxSelected>>", self.update_formula_info)

    def validate_numeric_input(self, value: str) -> bool:
        """Validate that input is a positive number with optional decimal."""
        if value == "":
            return True
        try:
            float(value)
            return float(value) > 0
        except ValueError:
            return False

    def validate_integer_input(self, value: str) -> bool:
        """Validate that input is a positive integer."""
        if value == "":
            return True
        try:
            int(value)
            return int(value) > 0
        except ValueError:
            return False

    def calculate_1rm(self) -> float | None:
        """
        Calculate 1RM using selected formula with input validation.

        Returns:
            Calculated 1RM or None if inputs are invalid.
        """
        try:
            weight = float(self.weight_entry.get())
            reps = int(self.rep_entry.get())

            if weight <= 0:
                raise ValueError("Weight must be greater than 0.")

            if reps <= 0:
                raise ValueError("Reps must be greater than 0.")

            if reps > 20:
                messagebox.warning(
                    "Warning",
                    "Formulas may be inaccurate for reps > 20.\n\n"
                    "1RM calculations are most reliable for 1-10 reps.",
                )

            formula_name = self.formula_var.get()
            formula = self.formulas[formula_name]
            return formula(weight, reps)

        except ValueError as e:
            self.result_value_label.config(text=f"Error: {str(e)}")
            return None
        except Exception as e:
            self.result_value_label.config(text=f"An error occurred: {str(e)}")
            return None

    def display_result(self) -> None:
        """Calculate 1RM and update result label with formatted value."""
        result = self.calculate_1rm()
        if result is not None:
            self.result_value_label.config(text=f"{result:.2f} kg")

    def update_formula_info(self, event=None) -> None:
        """Update the formula description label."""
        formula_name = self.formula_var.get()
        if formula_name == "Epley":
            desc = "Epley (w × (1 + r × 0.0333))"
        elif formula_name == "Brzycki":
            desc = "Brzycki (w ÷ (1.0278 - 0.0278 × r))"
        elif formula_name == "Lombardi":
            desc = "Lombardi (w × r^0.1)"
        elif formula_name == "Mayhew":
            desc = "Mayhew (100w ÷ (52.2 + 41.9e^(-0.055r)))"
        else:
            desc = formula_name

        self.formula_info_label.config(text=f"Formula: {desc}")

    def apply_dark_mode(self) -> None:
        """Apply dark mode color scheme."""
        self.bg_color = "#1E1E1E"
        self.text_color = "#FFFFFF"
        self.entry_bg = "#333333"
        self.button_bg = "#4CAF50"
        self.button_fg = "#FFFFFF"
        self.button_active_bg = "#45a049"
        self.window.configure(bg=self.bg_color)

    def apply_light_mode(self) -> None:
        """Apply light mode color scheme."""
        self.bg_color = "#F0F0F0"
        self.text_color = "#000000"
        self.entry_bg = "#FFFFFF"
        self.button_bg = "#4CAF50"
        self.button_fg = "#FFFFFF"
        self.button_active_bg = "#45a049"
        self.window.configure(bg=self.bg_color)

    def toggle_theme(self) -> None:
        """Toggle between dark and light modes."""
        self.is_dark_mode = not self.is_dark_mode

        if self.is_dark_mode:
            self.apply_dark_mode()
        else:
            self.apply_light_mode()

        # Reconfigure all widgets
        for widget in self.window.winfo_children():
            if isinstance(widget, tk.Label) or isinstance(widget, tk.Button):
                widget.configure(bg=self.bg_color, fg=self.text_color)
            elif isinstance(widget, tk.Entry):
                widget.configure(
                    bg=self.entry_bg,
                    fg=self.text_color,
                    insertbackground=self.text_color,
                )
            elif isinstance(widget, tk.Frame):
                widget.configure(bg=self.bg_color)
                for sub_widget in widget.winfo_children():
                    if isinstance(sub_widget, tk.Label) or isinstance(
                        sub_widget, tk.Button
                    ):
                        sub_widget.configure(bg=self.bg_color, fg=self.text_color)
                    elif isinstance(sub_widget, tk.Entry):
                        sub_widget.configure(
                            bg=self.entry_bg,
                            fg=self.text_color,
                            insertbackground=self.text_color,
                        )
                    elif isinstance(sub_widget, ttk.Combobox):
                        # ttk widgets require style configuration
                        style = ttk.Style()
                        if self.is_dark_mode:
                            style.configure(
                                "TCombobox",
                                fieldbackground=self.entry_bg,
                                background=self.bg_color,
                                foreground=self.text_color,
                            )
                        else:
                            style.configure(
                                "TCombobox",
                                fieldbackground=self.entry_bg,
                                background=self.bg_color,
                                foreground=self.text_color,
                            )

        # Update button colors
        for widget in self.window.winfo_children():
            if isinstance(widget, tk.Frame):
                for button in widget.winfo_children():
                    if isinstance(button, tk.Button):
                        button.configure(
                            bg=self.button_bg,
                            fg=self.button_fg,
                            activebackground=self.button_active_bg,
                        )

        # Update result label accent color
        self.result_value_label.configure(
            fg="#22A7F0" if self.is_dark_mode else "#1E88E5"
        )
        self.formula_info_label.configure(
            fg="#888888" if self.is_dark_mode else "#666666"
        )

    def run(self) -> None:
        """Start the application's main event loop."""
        self.window.mainloop()


if __name__ == "__main__":
    calculator = OneRepMaxCalculator()
    calculator.run()
