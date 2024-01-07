from word2number import w2n

# ? word2number then w2n then .word_to_num? So, library(bunch of modules) then module then method(function)???!
# return w2n.word_to_num(input_value)


# TODO: Instead of rounding at the destination, round at the source.
# Reason: As per the program need, I don't want a functionality to round or not round the number, based on the requirement, I always want to round the number.
#! Will see it tomorrow.


class DetermineSign:
    def __init__(self, num=None):
        if num is None:
            self.get_number()
        else:
            self.num = round(self.convert_to_float(num), 1)

    # TODO: Word2number

    # Need to further understand this.
    # ? NEED TO UNDERSTAND THIS. FOR SURETY.
    def convert_to_float(self, input_value):
        try:
            return float(input_value)
        except ValueError:
            try:
                return w2n.word_to_num(input_value)
            except ValueError:
                raise ValueError(
                    "Invalid input. Please enter a number or a word representing a number."
                )

    # Now use this in other methods.

    def get_number(self):
        self.input_value = format(float(input("Enter a number: ")), ".1f")
        self.num = round(self.convert_to_float(self.input_value), 1)
        return self.num
        # Do I want to return the self.num?
        # I think I have to just store it as it is.

    def determine_sign(self):
        if self.num > 0:
            return "Positive number"
        elif self.num < 0:
            return "Negative number"
        else:
            return "Zero"

    def __repr__(self):
        return f"Number: {self.num}, Sign: {self.determine_sign()}"


if __name__ == "__main__":
    number1 = DetermineSign()
    print(number1.determine_sign())


# !Incomplete.
