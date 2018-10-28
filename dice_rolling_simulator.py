"""
Dice Rolling Simulator
"""
# Made on May 27th, 2017
# Made by SlimxShadyx
# Editted by CaptMcTavish, June 17th, 2017
# Comments edits by SlimxShadyx, August 11th, 2017


import random


def start():
    """
    Our start function (What the user will first see when starting the program)
    """
    print "Welcome to dice rolling simulator: \nPress Enter to proceed"
    raw_input(">")

    # Starting our result function (The dice picker function)
    result()


def bye():
    """
    Our exit function
    (What the user will see when choosing to exit the program)
    """
    print "Thanks for using the Dice Rolling Simulator! Have a great day! =)"


def result():
    """
    Result function which is our dice chooser function
    """
    # user_dice_chooser  No idea how this got in here, thanks EroMonsterSanji.

    print "\r\nGreat! Begin by choosing a die! [6] [8] [12]?\r\n"
    user_dice_chooser = raw_input(">")

    user_dice_chooser = int(user_dice_chooser)

    # Below is the references to our dice functions (Below),
    # when the user chooses a dice.
    if user_dice_chooser == 6:
        dice6()

    elif user_dice_chooser == 8:
        dice8()

    elif user_dice_chooser == 12:
        dice12()

    # If the user doesn't choose an applicable option
    else:
        print "\r\nPlease choose one of the applicable options!\r\n"
        result()


def dice6():
    """
    Below are our dice functions.
    """
    # Getting a random number between 1 and 6 and printing it.
    dice_6 = random.randint(1, 6)
    print "\r\nYou rolled a " + str(dice_6) + "!\r\n"

    user_exit_checker()


def dice8():
    """
    Generate a dice with 8 sides.
    """
    dice_8 = random.randint(1, 8)
    print "\r\nYou rolled a " + str(dice_8) + "!"

    user_exit_checker()


def dice12():
    """
    Generate a dice with 12 sides.
    """
    dice_12 = random.randint(1, 12)
    print "\r\nYou rolled a " + str(dice_12) + "!"

    user_exit_checker()


def user_exit_checker():
    """
    Checking if the user would like to roll another die, or to exit the program
    """
    user_exit_checker_raw = raw_input(
        "\r\nIf you want to roll another die, type [roll]. "
        + "To exit, type [exit].\r\n?>")
    user_exit_checker_new = (user_exit_checker_raw.lower())
    if user_exit_checker_new == "roll":
        start()
    else:
        bye()


# Actually starting the program now.
start()
