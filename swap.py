class Swapper:
    """
    A class to perform swapping of two values.

    Methods:
    -------
    swap_tuple_unpacking(self):
        Swaps the values of x and y using a tuple unpacking method.
    
    swap_temp_variable(self):
        Swaps the values of x and y using a temporary variable.
    
    swap_arithmetic_operations(self):
        Swaps the values of x and y using arithmetic operations.

    """

    def __init__(self, x, y):
        """
        Initialize the Swapper class with two values.

        Parameters:
        ----------
        x : int
            The first value to be swapped.
        y : int
            The second value to be swapped.

        """
        if not isinstance(x, (int, float)) or not isinstance(y, (float, int)):
            raise ValueError("Both x and y should be integers.")
        
        self.x = x
        self.y = y

    def display_values(self, message):
        print(f"{message} x: {self.x}, y: {self.y}")

    def swap_tuple_unpacking(self):
        """
        Swaps the values of x and y using a tuple unpacking method.

        """
        self.display_values("Before swapping")
        self.x, self.y = self.y, self.x
        self.display_values("After swapping")

    def swap_temp_variable(self):
        """
        Swaps the values of x and y using a temporary variable.

        """
        self.display_values("Before swapping")
        temp = self.x
        self.x = self.y
        self.y = temp
        self.display_values("After swapping")

    def swap_arithmetic_operations(self):
        """
        Swaps the values of x and y using arithmetic operations.

        """
        self.display_values("Before swapping")
        self.x = self.x - self.y
        self.y = self.x + self.y
        self.x = self.y - self.x
        self.display_values("After swapping")


print("Example 1:")
swapper1 = Swapper(5, 10)
swapper1.swap_tuple_unpacking()
print()

print("Example 2:")
swapper2 = Swapper(100, 200)
swapper2.swap_temp_variable()
print()
