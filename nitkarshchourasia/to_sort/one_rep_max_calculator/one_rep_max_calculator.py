class OneRepMaxCalculator:
    """
    A class to calculate the one-repetition maximum (1RM) for a weightlifting exercise.
    """

    def __init__(self):
        """
        Initializes the OneRepMaxCalculator with default values.
        """
        self.weight_lifted = 0
        self.reps_performed = 0

    def get_user_input(self):
        """
        Prompts the user to enter the weight lifted and the number of reps performed.
        """
        self.weight_lifted = int(input("Enter the weight you lifted (in kg): "))
        self.reps_performed = int(input("Enter the number of reps you performed: "))

    def calculate_one_rep_max(self):
        """
        Calculates the one-rep max based on the Epley formula.
        """
        return (self.weight_lifted * self.reps_performed * 0.0333) + self.weight_lifted

    def display_one_rep_max(self):
        """
        Displays the calculated one-rep max.
        """
        one_rep_max = self.calculate_one_rep_max()
        print(f"Your estimated one-rep max (1RM) is: {one_rep_max} kg")


def main():
    """
    The main function that creates an instance of OneRepMaxCalculator and uses it to get user input,
    calculate the one-rep max, and display the result.
    """
    calculator = OneRepMaxCalculator()
    calculator.get_user_input()
    calculator.display_one_rep_max()


if __name__ == "__main__":
    main()
