import random

class Die:
    """
    A class used to represent a multi-sided die.
    
    Attributes:
        sides (int): The number of sides on the die (default is 6).
    """

    def __init__(self, sides=6):
        """Initializes the die. Defaults to 6 sides if no value is provided."""
        self.sides = 6  # Internal default
        self.set_sides(sides)

    def set_sides(self, num_sides):
        """
        Validates and sets the number of sides. 
        A physical die must have at least 4 sides.
        """
        if isinstance(num_sides, int) and num_sides >= 4:
            if num_sides != self.sides:
                print(f"Changing sides from {self.sides} to {num_sides}!")
            else:
                print(f"Sides already set to {num_sides}.")
            self.sides = num_sides
        else:
            print(f"Invalid input: {num_sides}. Keeping current value: {self.sides}")

    def roll(self):
        """Returns a random integer between 1 and the number of sides."""
        return random.randint(1, self.sides)

# --- Example Usage ---
if __name__ == "__main__":
    d1 = Die(4)  # Initialize directly with 4 sides
    d2 = Die(12) # A Dungeons & Dragons classic
    
    print(f"Roll Result: D{d1.sides} -> {d1.roll()}, D{d2.sides} -> {d2.roll()}")
