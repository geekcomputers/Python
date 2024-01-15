__author__ = "Nitkarsh Chourasia"
__author_GitHub_profile__ = "https://github.com/NitkarshChourasia"
__author_email_address__ = "playnitkarsh@gmal.com"
__created_on__ = "10/10/2021"
__last_updated__ = "10/10/2021"

from word2number import w2n


def convert_words_to_number(word_str):
    """
    Convert a string containing number words to an integer.

    Args:
    - word_str (str): Input string with number words.

    Returns:
    - int: Numeric equivalent of the input string.
    """
    numeric_result = w2n.word_to_num(word_str)
    return numeric_result


# Example usage:
number_str = "two hundred fifteen"
result = convert_words_to_number(number_str)
print(result)  # Output: 215


class Square:
    def __init__(self, side=None):
        if side is None:
            self.ask_side()
        # else:
        #     self.side = float(side)
        else:
            if not isinstance(side, (int, float)):
                try:
                    side = float(side)
                except ValueError:
                    # return "Invalid input for side."
                    raise ValueError("Invalid input for side.")
            else:
                self.side = float(side)
        # Check if the result is a float and remove unnecessary zeros

        self.calculate_square()
        self.truncate_decimals()

    # If ask side or input directly into the square.
    # That can be done?
    def calculate_square(self):
        self.area = self.side * self.side
        return self.area

    # Want to add a while loop asking for the input.
    # Also have an option to ask the user in true mode or in repeat mode.
    def ask_side(self):
        # if true bool then while if int or float then for loop.
        # I will have to learn inheritance and polymorphism.
        condition = 3
        # condition = True
        if condition == True and isinstance(condition, bool):
            while condition:
                n = input("Enter the side of the square: ")
                self.side = float(n)
        elif isinstance(condition, (int, float)):
            for i in range(_=condition):
                n = input("Enter the side of the square: ")
                self.side = float(n)
        # n = input("Enter the side of the square: ")
        # self.side = float(n)
        # return

    def truncate_decimals(self):
        return (
            f"{self.area:.10f}".rstrip("0").rstrip(".")
            if "." in str(self.area)
            else self.area
        )

    # Prettifying the output.

    def calculate_perimeter(self):
        return 4 * self.side

    def calculate_perimeter_prettify(self):
        return f"The perimeter of the square is {self.calculate_perimeter()}."

    def calculate_area_prettify(self):
        return f"The area of the square is {self.area}."

    def truncate_decimals_prettify(self):
        return f"The area of the square is {self.truncate_decimals()}."


if __name__ == "__main__":
    output_one = Square()
    truncated_area = output_one.truncate_decimals()
    # print(output_one.truncate_decimals())
    print(truncated_area)


# add a while loop to keep asking for the user input.
# also make sure to add a about menu to input a while loop in tkinter app.

# It can use a beautiful GUI also.
# Even validation is left.
# What if string is provided in number? Then?
# What if chars are provided. Then?
# What if a negative number is provided? Then?
# What if a number is provided in alphabets characters? Then?
# Can it a single method have more object in it?

# Also need to perform testing on it.
# EXTREME FORM OF TESTING NEED TO BE PERFORMED ON IT.
# Documentation is also needed.
# Comments are also needed.
# TYPE hints are also needed.

# README.md file is also needed.
## Which will explain the whole project.
### Like how to use the application.
### List down the features in explicit detail.
### How to use different methods and classes.
### It will also a image of the project in working state.
### It will also have a video to the project in working state.

# It should also have .exe and linux executable file.
# It should also be installable into Windows(x86) system and if possible into Linux system also.
