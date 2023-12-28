# # Returns the area of the square with given sides
# n = input("Enter the side of the square: ")  # Side length should be given in input
# side = float(n)
# area = side * side  # calculate area
# print("Area of the given square is ", area)


class Square:
    def __init__(self, side=None):
        if side is None:
            self.ask_side()
        else:
            self.side = float(side)

        self.square()
        self.truncate_decimals()

    # If ask side or input directly into the square.
    # That can be done?
    def square(self):
        self.area = self.side * self.side
        return self.area

    def ask_side(self):
        n = input("Enter the side of the square: ")
        self.side = float(n)
        # return

    def truncate_decimals(self):
        return (
            f"{self.area:.10f}".rstrip("0").rstrip(".")
            if "." in str(self.area)
            else self.area
        )


# Even validation is left.
# What if string is provided in number? Then?
# What if chars are provided. Then?
# What if a negative number is provided? Then?
# What if a number is provided in alphabets characters? Then?
# Can it a single method have more object in it?

if __name__ == "__main__":
    output_one = Square()
    truncated_area = output_one.truncate_decimals()
    # print(output_one.truncate_decimals())
    print(truncated_area)


# It can use a beautiful GUI also.
