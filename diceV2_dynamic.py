import random


# Class that that holds dice-functions. You can set the amount of sides and roll with each dice object.
class Dice:
    def __init__(self):
        self.sideCount = 6

    def setSides(self, sides):
        if sides > 3:
            self.sides = sides
        else:
            print(
                "This absolutely shouldn't ever happen. The programmer sucks or someone "
                "has tweaked with code they weren't supposed to touch!"
            )

    def roll(self):
        return random.randint(1, self.sides)


# =====================================================================


# Checks to make sure that the input is actually an integer.
# This implementation can be improved greatly of course.
def checkInput(sides):
    try:
        if int(sides) != 0:
            if (
                float(sides) % int(sides) == 0
            ):  # excludes the possibility of inputted floats being rounded.
                return int(sides)
        else:
            return int(sides)

    except ValueError:
        print("Invalid input!")
        return None


# Picks a number that is at least of a certain size.
# That means in this program, the dices being possible to use in 3 dimensional space.
def pickNumber(item, question_string, lower_limit):
    while True:
        item = input(question_string)
        item = checkInput(item)
        if type(item) == int:
            if item <= lower_limit:
                print("Input too low!")
                continue
            else:
                return item


# Main-function of the program that sets up the dices for the user as they want them.
def getDices():
    dices = []
    sides = None
    diceAmount = None
    sideLowerLimit = 3  # Do Not Touch!
    diceLowerLimit = 1  # Do Not Touch!

    sides = pickNumber(sides, "How many sides will the dices have?: ", sideLowerLimit)
    diceAmount = pickNumber(
        diceAmount, "How many dices will do you want?: ", diceLowerLimit
    )

    for i in range(0, diceAmount):
        d = Dice()
        d.setSides(sides)
        dices.append(d)

    return dices


# =================================================================
# Output section.


def output():
    dices = getDices()
    input("Do you wanna roll? press enter")
    cont = True
    while cont:
        rollOutput = ""
        for dice in dices:
            rollOutput = rollOutput + str(dice.roll()) + ", "
        rollOutput = rollOutput[:-2]
        print(rollOutput)

        print("do you want to roll again?")
        ans = input("press enter to continue, and [exit] to exit")
        if ans == "exit":
            cont = False


if __name__ == "__main__":
    output()
