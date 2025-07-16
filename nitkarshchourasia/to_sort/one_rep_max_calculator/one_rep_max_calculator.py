class OneRepMaxCalculator:
    """
    A class to calculate the one-repetition maximum (1RM) for a weightlifting exercise.
    """

    def __init__(self) -> None:
        """
        Initializes the OneRepMaxCalculator with default values.
        """
        self.weight_lifted: int = 0
        self.reps_performed: int = 0
        self.formulas = {
            "epley": lambda w, r: w * (1 + r * 0.0333),
            "brzycki": lambda w, r: w / (1.0278 - 0.0278 * r),
            "lombardi": lambda w, r: w * (r ** 0.1),
            "mayhew": lambda w, r: (100 * w) / (52.2 + 41.9 * (2.71828 ** (-0.055 * r))),
        }

    def get_user_input(self) -> None:
        """
        Prompts the user to enter the weight lifted and the number of reps performed.
        Validates inputs to ensure they are positive integers.
        """
        while True:
            try:
                self.weight_lifted = int(input("Enter the weight you lifted (in kg): "))
                self.reps_performed = int(input("Enter the number of reps you performed: "))
                if self.weight_lifted <= 0 or self.reps_performed <= 0:
                    raise ValueError("Weight and reps must be positive numbers.")
                break
            except ValueError as e:
                print(f"Error: {e}. Please try again.")

    def calculate_one_rep_max(self, formula: str = "epley") -> float:
        """
        Calculates the one-rep max using the specified formula.
        
        Args:
            formula: The name of the formula to use (default: "epley").
            
        Returns:
            The estimated one-rep max weight in kilograms.
        """
        if formula not in self.formulas:
            raise ValueError(f"Invalid formula. Available options: {', '.join(self.formulas.keys())}")
        return self.formulas[formula](self.weight_lifted, self.reps_performed)

    def display_one_rep_max(self, formula: str = "epley") -> None:
        """
        Displays the calculated one-rep max with two decimal places.
        
        Args:
            formula: The name of the formula to use (default: "epley").
        """
        one_rep_max = self.calculate_one_rep_max(formula)
        print(f"Your estimated one-rep max (1RM) using {formula.title()} formula is: {one_rep_max:.2f} kg")


def main() -> None:
    """
    The main function that creates an instance of OneRepMaxCalculator and uses it to get user input,
    calculate the one-rep max using the Epley formula, and display the result.
    """
    calculator = OneRepMaxCalculator()
    calculator.get_user_input()
    calculator.display_one_rep_max()


if __name__ == "__main__":
    main()