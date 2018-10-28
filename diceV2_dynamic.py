"""
Version 2 of the dice script.
"""
import random


class Dice(object):
    """
    Class that that holds dice-functions.
    You can set the amount of sides and roll with each dice object.
    """
    def __init__(self):
        self.sides = 6

    def set_sides(self, sides):
        """
        Setter method for the number of sides on the dice.
        """
        if sides > 3:
            self.sides = sides
        else:
            print("This absolutely shouldn't ever happen." +
                  "The programmer sucks or someone " +
                  "has tweaked with code they weren't supposed to touch!")

    def roll(self):
        """
        Getter method for a random side value on the dice.
        """
        return random.randint(1, self.sides)


# =====================================================================


def check_input(sides):
    """
    Checks to make sure that the input is actually an integer.
    This implementation can be improved greatly of course.
    """
    try:
        if int(sides) != 0:
            # excludes the possibility of inputted floats being rounded.
            if float(sides) % int(sides) == 0:
                return int(sides)
        else:
            return int(sides)

    except ValueError:
        print "Invalid input!"
        return None


def pick_number(item, question_string, lower_limit):
    """
    Picks a number that is at least of a certain size.
    That means in this program, the dices being possible
    to use in 3 dimensional space.
    """
    while True:
        item = input(question_string)
        item = check_input(item)
        if isinstance(item) == int:
            if item <= lower_limit:
                print "Input too low!"
                continue
            else:
                return item


def get_dices():
    """
    Main-function of the program that sets up the dices for
    the user as they want them.
    """
    all_dice = []
    sides = None
    dice_amount = None
    side_lower_limit = 3  # Do Not Touch!
    dice_lower_limit = 1  # Do Not Touch!

    sides = pick_number(sides, "How many sides will the dices have?: ",
                        side_lower_limit)
    dice_amount = pick_number(dice_amount,
                              "How many dices will do you want?: ",
                              dice_lower_limit)

    for _ in range(0, dice_amount):
        die = Dice()
        die.set_sides(sides)
        all_dice.append(die)

    return all_dice


DICES = get_dices()
# =================================================================
# Output section.


ROLL_OUTPUT = ""

for dice in DICES:
    ROLL_OUTPUT = ROLL_OUTPUT + str(dice.roll()) + ", "

ROLL_OUTPUT = ROLL_OUTPUT[:-2]
print ROLL_OUTPUT
