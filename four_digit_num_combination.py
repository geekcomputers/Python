""" small script to learn how to print out all 4-digit num"""


# ALL the combinations of 4 digit combo
def four_digit_combinations():
    """ print out all 4-digit numbers in old way"""
    numbers = []
    for code in range(10000):
        code = str(code).zfill(4)
        print(code)
        numbers.append(code)


# Same as above but more pythonic
def one_line_combinations():
    """ print out all 4-digit numbers """
    numbers = [str(i).zfill(4) for i in range(10000)]
    print(numbers)
